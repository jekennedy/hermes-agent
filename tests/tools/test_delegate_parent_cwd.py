"""Delegation must inherit the terminal's workspace, not the configured checkout.

Gateway routing keys and agent task IDs can differ. Exercise real Git worktrees,
terminal execution, context propagation and file resolution without an LLM call.
"""
import json
import shlex
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import terminal_tool as tt
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.delegate_tool_child_run import _ChildRun
from tools.file_tools_paths import _resolve_path_for_task
from tools.terminal_scope import reset_terminal_scope, set_terminal_scope


def _git(repo, *args):
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, text=True, capture_output=True,
    ).stdout.strip()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    repo = tmp_path / 'main'
    repo.mkdir()
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 'test@example.test')
    _git(repo, 'config', 'user.name', 'Test')
    (repo / 'base.txt').write_text('main\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-qm', 'base')
    feature = tmp_path / 'feature'
    _git(repo, 'worktree', 'add', '-qb', 'feature', str(feature))
    (feature / 'feature.txt').write_text('approved feature baseline\n')
    _git(feature, 'add', '.')
    _git(feature, 'commit', '-qm', 'feature baseline')
    monkeypatch.setattr(tt, '_session_cwd', {})
    monkeypatch.setattr(tt, '_container_aliases', {})
    monkeypatch.setattr(tt, '_task_env_overrides', {})
    token = set_terminal_scope({'TERMINAL_ENV': 'local', 'TERMINAL_CWD': str(repo)})
    try:
        yield repo, feature
    finally:
        tt.cleanup_all_environments()
        reset_terminal_scope(token)


@pytest.mark.parametrize('isolation', [False, True])
@pytest.mark.parametrize('routing_key', ['', 'internal-parent', 'discord:thread:parent'])
def test_child_inherits_live_parent_workspace(workspace, monkeypatch, isolation, routing_key):
    repo, feature = workspace
    monkeypatch.setattr('tools.delegate_tool._get_worktree_isolation', lambda: isolation)
    task_id = 'internal-parent'
    token = set_current_session_key(routing_key)
    try:
        # The actual terminal writes the record under the routing key, if set.
        result = json.loads(tt.terminal_tool(command=f'cd {shlex.quote(str(feature))} && pwd', task_id=task_id))
        assert result['exit_code'] == 0, result
        assert Path(result['output'].strip()).resolve() == feature.resolve()
        key = routing_key or task_id
        assert tt.get_session_cwd(key) == str(feature)
        run = _ChildRun(SimpleNamespace(), SimpleNamespace(_current_task_id=task_id),
                        0, 'Inspect the feature', 'child', None)
        run.seed_workspace()
        expected = str(feature)
        if isolation:
            assert run.worktree_info is not None
            expected = run.worktree_info['path']
            assert run.worktree_info['base_commit'] == _git(feature, 'rev-parse', 'HEAD')
            assert _git(expected, 'rev-parse', 'HEAD') != _git(repo, 'rev-parse', 'HEAD')
        assert tt.get_session_cwd(run.child_task_id) == expected
        assert _resolve_path_for_task('feature.txt', task_id=run.child_task_id) == Path(expected) / 'feature.txt'
        assert tt.get_session_cwd(key) == str(feature)
        # Workspace identity must not replace container/process ownership identity.
        assert run.parent_task_id == task_id
        assert tt._container_aliases[run.child_task_id] == task_id
    finally:
        reset_current_session_key(token)


@pytest.mark.parametrize('isolation', [False, True])
def test_child_inherits_workspace_override_before_first_parent_terminal(workspace, monkeypatch, isolation):
    repo, feature = workspace
    monkeypatch.setattr('tools.delegate_tool._get_worktree_isolation', lambda: isolation)
    task_id = 'internal-parent'
    routing_key = 'discord:thread:parent'
    tt.register_task_env_overrides(task_id, {'cwd': str(feature)})
    token = set_current_session_key(routing_key)
    try:
        assert tt.get_session_cwd(routing_key) is None
        run = _ChildRun(SimpleNamespace(), SimpleNamespace(_current_task_id=task_id),
                        0, 'Inspect the feature', 'child', None)
        run.seed_workspace()
        expected = str(feature)
        if isolation:
            assert run.worktree_info is not None
            assert run.worktree_info['base_commit'] == _git(feature, 'rev-parse', 'HEAD')
            expected = run.worktree_info['path']
        assert tt.get_session_cwd(run.child_task_id) == expected
        assert _resolve_path_for_task('feature.txt', task_id=run.child_task_id) == Path(expected) / 'feature.txt'
        assert tt.get_session_cwd(task_id) == str(feature)
        assert tt.get_session_cwd(routing_key) is None
        assert tt._container_aliases[run.child_task_id] == task_id
    finally:
        reset_current_session_key(token)


@pytest.mark.parametrize('after_cd', [False, True])
def test_constructed_child_prompt_uses_live_parent_project(workspace, monkeypatch, after_cd):
    from run_agent import AIAgent
    from tools import delegate_tool as dt

    repo, feature = workspace
    (repo / 'AGENTS.md').write_text('MAIN_PROJECT_ONLY_INSTRUCTIONS\n')
    (feature / 'AGENTS.md').write_text('FEATURE_PROJECT_ONLY_INSTRUCTIONS\n')
    monkeypatch.setattr(dt, '_load_config', lambda: {})
    # Metadata is unrelated to workspace identity; keep real construction offline.
    monkeypatch.setattr('agent.model_metadata.fetch_model_metadata', lambda **kwargs: {})
    parent = AIAgent(
        session_id='parent-session', api_key='not-a-real-key',
        base_url='https://openrouter.ai/api/v1', provider='openrouter',
        api_mode='chat_completions', model='anthropic/claude-sonnet-4.6',
        platform='cli', quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False, enabled_toolsets=['file'],
    )
    parent._current_task_id = 'internal-parent'
    # Leave a different raw record behind after cd to prove routed-cwd precedence.
    tt.register_task_env_overrides(parent._current_task_id, {'cwd': str(repo if after_cd else feature)})
    token = set_current_session_key('discord:thread:parent')
    child = None
    try:
        if after_cd:
            result = json.loads(tt.terminal_tool(
                command=f'cd {shlex.quote(str(feature))}', task_id=parent._current_task_id,
            ))
            assert result['exit_code'] == 0, result
        child = dt._build_child_agent(
            task_index=0, goal='Inspect the feature', context=None, toolsets=['file'],
            model=None, max_iterations=4, task_count=1, parent_agent=parent,
        )
        prompt = child.ephemeral_system_prompt
        assert f'WORKSPACE PATH:\n{feature}\n' in prompt
        assert 'FEATURE_PROJECT_ONLY_INSTRUCTIONS' in prompt
        assert 'MAIN_PROJECT_ONLY_INSTRUCTIONS' not in prompt
        assert child.skip_context_files is True
    finally:
        if child is not None:
            child.close()
        parent.close()
        reset_current_session_key(token)


@pytest.mark.parametrize('backend', ['local', 'ssh', 'docker'])
def test_workspace_hint_preserves_host_validation_and_remote_behavior(workspace, backend):
    from tools.delegate_tool_progress import _resolve_workspace_hint

    repo, feature = workspace
    parent = SimpleNamespace(_current_task_id='internal-parent')
    # A coincidentally existing host path must not make a remote cwd a local hint.
    live_cwd = str(feature / 'missing') if backend == 'local' else str(feature)
    token = set_terminal_scope({'TERMINAL_ENV': backend, 'TERMINAL_CWD': str(repo)})
    routing_token = set_current_session_key('discord:thread:parent')
    try:
        tt.record_session_cwd('discord:thread:parent', live_cwd)
        assert _resolve_workspace_hint(parent) == str(repo)
    finally:
        reset_current_session_key(routing_token)
        reset_terminal_scope(token)


@pytest.mark.parametrize('notification', ['complete', 'watch', 'yield'])
def test_child_background_keeps_parent_routing_and_child_cwd(workspace, monkeypatch, notification):
    from agent.delegation_context import delegated_child_context
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import process_registry as pr
    from tools.approval_context import get_current_session_key
    from tools.interrupt import consume_yield, request_yield

    repo, feature = workspace
    registry = pr.ProcessRegistry()
    monkeypatch.setattr(pr, 'process_registry', registry)
    child_task = 'child-task'
    routing_key = 'discord:thread:parent'
    tt.record_session_cwd(routing_key, str(repo))
    tt.record_session_cwd(child_task, str(feature))
    tt.register_container_alias(child_task, 'internal-parent')
    token = set_current_session_key(routing_key)
    session_tokens = set_session_vars(platform='discord', chat_id='test-chat',
                                      thread_id='test-thread', session_key=routing_key,
                                      session_id='parent-session')
    try:
        with delegated_child_context():
            if notification == 'yield':
                request_yield(threading.get_ident())
            result = json.loads(tt.terminal_tool(
                command='pwd; sleep 2' if notification == 'yield' else 'pwd',
                task_id=child_task, background=notification != 'yield',
                notify_on_complete=notification == 'complete',
                watch_patterns=['feature'] if notification == 'watch' else None,
            ))
            if notification == 'yield':
                assert result['status'] == 'yielded_to_background', result
            else:
                assert result['exit_code'] == 0, result
            completed = registry.wait(result['session_id'], timeout=10)
            assert completed['exit_code'] == 0, completed
            process = registry.get(result['session_id'])
            assert process is not None and process.cwd is not None
            assert process.owner_task_id == child_task
            assert process.session_key == routing_key
            assert Path(process.cwd).resolve() == feature.resolve()
            assert str(feature) in completed['output']
            assert process.watcher_chat_id == 'test-chat'
            assert process.watcher_thread_id == 'test-thread'
            if notification in ('complete', 'yield'):
                assert registry.pending_watchers[0]['session_key'] == routing_key
            assert get_current_session_key() == routing_key
            assert tt.get_session_cwd(routing_key) == str(repo)
            assert tt.get_session_cwd(child_task) == str(feature)
    finally:
        consume_yield(threading.get_ident())
        clear_session_vars(session_tokens)
        reset_current_session_key(token)


def test_child_terminal_and_nested_delegation_keep_their_own_workspace(workspace, monkeypatch):
    repo, feature = workspace
    monkeypatch.setattr('tools.delegate_tool._get_worktree_isolation', lambda: True)
    routing_key = 'discord:thread:parent'
    task_id = 'internal-parent'
    token = set_current_session_key(routing_key)
    seen = {}

    class Child:
        session_id = 'child-session'

        def run_conversation(self, *, task_id, **kwargs):
            from tools.approval_context import get_current_session_key

            # Real worker inherits approval routing, but cwd must remain child-local.
            seen['approval_key'] = get_current_session_key()
            seen['pwd'] = json.loads(tt.terminal_tool(command='pwd', task_id=task_id))
            own_cwd = tt.get_session_cwd(task_id)
            assert own_cwd is not None
            own = Path(own_cwd)
            (own / 'child-only.txt').write_text('child commit\n')
            _git(own, 'add', 'child-only.txt')
            _git(own, 'commit', '-qm', 'child commit')
            subdir = own / 'subdir'
            subdir.mkdir()
            changed = json.loads(tt.terminal_tool(
                command=f'cd {shlex.quote(str(subdir))}', task_id=task_id,
            ))
            assert changed['exit_code'] == 0, changed
            persisted = json.loads(tt.terminal_tool(command='pwd', task_id=task_id))
            assert persisted['exit_code'] == 0, persisted
            assert Path(persisted['output'].strip()).resolve() == subdir.resolve()
            assert _resolve_path_for_task('../child-only.txt', task_id=task_id) == own / 'child-only.txt'
            nested = _ChildRun(SimpleNamespace(), SimpleNamespace(_current_task_id=task_id),
                               0, 'Inspect child commit', 'grandchild', None)
            nested.seed_workspace()
            seen['nested'] = nested.worktree_info
            seen['child_head'] = _git(own, 'rev-parse', 'HEAD')
            seen['file'] = _resolve_path_for_task('../child-only.txt', task_id=task_id)
            return {'final_response': 'ok', 'completed': True}

    try:
        result = json.loads(tt.terminal_tool(command=f'cd {shlex.quote(str(feature))}', task_id=task_id))
        assert result['exit_code'] == 0, result
        run = _ChildRun(Child(), SimpleNamespace(_current_task_id=task_id),
                        0, 'Inspect the feature', 'child', None)
        run.seed_workspace()
        result, failure, deferred = run.await_child()
        assert failure is None, failure
        assert not deferred
        assert result is not None and result['completed']
        assert run.worktree_info is not None
        assert seen['approval_key'] == routing_key
        assert seen['pwd']['exit_code'] == 0, seen['pwd']
        assert Path(seen['pwd']['output'].strip()).resolve() == Path(run.worktree_info['path']).resolve()
        assert seen['nested']['base_commit'] == seen['child_head']
        assert seen['file'] == Path(run.worktree_info['path']) / 'child-only.txt'
        assert tt.get_session_cwd(routing_key) == str(feature)
        assert not (feature / 'child-only.txt').exists()
        assert not (repo / 'child-only.txt').exists()
    finally:
        reset_current_session_key(token)
