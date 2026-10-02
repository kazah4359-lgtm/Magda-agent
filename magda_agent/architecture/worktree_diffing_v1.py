"""
Worktree State Diffing Utility V1

Provides state diffing and isolation leak analysis across subagent Git worktrees
prior to state merges. Inspired by Claude Agent SDK Agent Teams.
"""

from dataclasses import dataclass, field
from enum import Enum
import fnmatch
import os
import subprocess
from typing import Dict, List, Optional, Any, Set


class LeakType(str, Enum):
    """Types of isolation leaks or cross-contamination detected in worktrees."""
    OUT_OF_BOUNDS_PATH = "out_of_bounds_path"
    SCOPE_VIOLATION = "scope_violation"
    PROTECTED_PATH = "protected_path"
    CROSS_WORKTREE_OVERLAP = "cross_worktree_overlap"
    EXTERNAL_SYMLINK = "external_symlink"


class WorktreeContaminationError(Exception):
    """Raised when isolation leaks or cross-contamination are detected prior to merge."""
    pass


@dataclass
class IsolationLeak:
    """Represents a single identified isolation leak or cross-contamination event."""
    agent_id: str
    leak_type: LeakType
    filepath: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorktreeDiffResult:
    """Result of state diffing and leak analysis for a worktree."""
    agent_id: str
    worktree_path: str
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    leaks: List[IsolationLeak] = field(default_factory=list)
    overlapping_files: List[str] = field(default_factory=list)

    @property
    def is_contaminated(self) -> bool:
        """Returns True if any isolation leaks or overlaps were detected."""
        return len(self.leaks) > 0


class WorktreeStateDifferV1:
    """
    Evaluates state diffs and isolation leaks across subagent git worktrees
    prior to merging state back into the main repository.
    """

    DEFAULT_PROTECTED_PATTERNS = [
        ".git/*",
        ".git",
        ".env*",
        "*secret*",
        ".github/*",
        "*.pem",
        "*.key"
    ]

    def __init__(self, protected_patterns: Optional[List[str]] = None) -> None:
        """
        Initialize the WorktreeStateDifferV1 instance.

        Args:
            protected_patterns (Optional[List[str]]): List of glob patterns for protected paths.
        """
        self.protected_patterns = protected_patterns if protected_patterns is not None else list(self.DEFAULT_PROTECTED_PATTERNS)

    def get_worktree_changes(self, worktree_path: str) -> Dict[str, List[str]]:
        """
        Retrieves modified, untracked, and deleted files in a worktree path.
        Attempts `git status --porcelain` if `worktree_path` is a git repository or worktree,
        and falls back to filesystem inspection.

        Args:
            worktree_path (str): The absolute or relative path to the worktree.

        Returns:
            Dict[str, List[str]]: Dictionary with 'modified', 'untracked', 'deleted' file lists.
        """
        path = os.path.abspath(worktree_path)
        modified: List[str] = []
        untracked: List[str] = []
        deleted: List[str] = []

        if not os.path.exists(path):
            return {"modified": [], "untracked": [], "deleted": []}

        try:
            cmd = ["git", "status", "--porcelain"]
            proc = subprocess.run(
                cmd,
                cwd=path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if proc.returncode == 0 and proc.stdout:
                for line in proc.stdout.splitlines():
                    if len(line) < 4:
                        continue
                    status_code = line[:2]
                    filepath = line[3:].strip()

                    # Handle quoted git paths if present
                    if filepath.startswith('"') and filepath.endswith('"'):
                        filepath = filepath[1:-1]

                    if status_code in ("??", " A", "A "):
                        untracked.append(filepath)
                    elif status_code in (" D", "D "):
                        deleted.append(filepath)
                    else:
                        modified.append(filepath)
                return {
                    "modified": sorted(list(set(modified))),
                    "untracked": sorted(list(set(untracked))),
                    "deleted": sorted(list(set(deleted)))
                }
        except Exception:
            pass

        # Fallback filesystem inspection if not a git worktree or if git fails
        for root, _, files in os.walk(path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), path)
                untracked.append(rel_path)

        return {
            "modified": sorted(list(set(modified))),
            "untracked": sorted(list(set(untracked))),
            "deleted": sorted(list(set(deleted)))
        }

    def evaluate_worktree(
        self,
        worktree_path: str,
        agent_id: str = "agent",
        allowed_paths: Optional[List[str]] = None
    ) -> WorktreeDiffResult:
        """
        Evaluates a single worktree for isolation leaks, path escapes, protected pattern matches,
        and scope violations.

        Args:
            worktree_path (str): Path to the worktree directory.
            agent_id (str): Identifier of the subagent operating in the worktree.
            allowed_paths (Optional[List[str]]): List of relative paths/glob patterns the agent is permitted to touch.

        Returns:
            WorktreeDiffResult: Result object summarizing changes and identified leaks.
        """
        abs_worktree = os.path.abspath(worktree_path)
        changes = self.get_worktree_changes(abs_worktree)

        modified = changes["modified"]
        untracked = changes["untracked"]
        deleted = changes["deleted"]

        all_changed = sorted(list(set(modified + untracked + deleted)))
        leaks: List[IsolationLeak] = []

        for rel_file in all_changed:
            target_path = os.path.normpath(os.path.join(abs_worktree, rel_file))

            # 1. Path traversal / Out-of-bounds check
            if not target_path.startswith(abs_worktree + os.sep) and target_path != abs_worktree:
                leaks.append(
                    IsolationLeak(
                        agent_id=agent_id,
                        leak_type=LeakType.OUT_OF_BOUNDS_PATH,
                        filepath=rel_file,
                        reason=f"Path '{rel_file}' escapes worktree root '{abs_worktree}'."
                    )
                )
                continue

            # 2. Symlink escape check
            full_file_path = os.path.join(abs_worktree, rel_file)
            if os.path.islink(full_file_path):
                real_target = os.path.realpath(full_file_path)
                if not real_target.startswith(abs_worktree + os.sep) and real_target != abs_worktree:
                    leaks.append(
                        IsolationLeak(
                            agent_id=agent_id,
                            leak_type=LeakType.EXTERNAL_SYMLINK,
                            filepath=rel_file,
                            reason=f"Symlink '{rel_file}' points outside worktree to '{real_target}'."
                        )
                    )

            # 3. Protected path check
            if self._is_protected(rel_file):
                leaks.append(
                    IsolationLeak(
                        agent_id=agent_id,
                        leak_type=LeakType.PROTECTED_PATH,
                        filepath=rel_file,
                        reason=f"File '{rel_file}' matches protected pattern."
                    )
                )

            # 4. Scope violation check (if allowed_paths specified)
            if allowed_paths is not None and not self._is_allowed(rel_file, allowed_paths):
                leaks.append(
                    IsolationLeak(
                        agent_id=agent_id,
                        leak_type=LeakType.SCOPE_VIOLATION,
                        filepath=rel_file,
                        reason=f"File '{rel_file}' is outside agent's assigned allowed paths."
                    )
                )

        return WorktreeDiffResult(
            agent_id=agent_id,
            worktree_path=abs_worktree,
            modified_files=modified,
            untracked_files=untracked,
            deleted_files=deleted,
            leaks=leaks
        )

    def evaluate_team_worktrees(
        self,
        worktree_map: Dict[str, str],
        allowed_paths_map: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, WorktreeDiffResult]:
        """
        Evaluates multiple agent worktrees simultaneously and identifies cross-worktree overlaps.

        Args:
            worktree_map (Dict[str, str]): Mapping of agent_id -> worktree_path.
            allowed_paths_map (Optional[Dict[str, List[str]]]): Mapping of agent_id -> allowed_paths.

        Returns:
            Dict[str, WorktreeDiffResult]: Results for each agent including cross-contamination analysis.
        """
        results: Dict[str, WorktreeDiffResult] = {}
        file_to_agents: Dict[str, Set[str]] = {}

        allowed_paths_map = allowed_paths_map or {}

        # First pass: Evaluate each worktree individually
        for agent_id, wt_path in worktree_map.items():
            allowed = allowed_paths_map.get(agent_id)
            res = self.evaluate_worktree(worktree_path=wt_path, agent_id=agent_id, allowed_paths=allowed)
            results[agent_id] = res

            # Track files changed across worktrees
            all_files = set(res.modified_files + res.untracked_files + res.deleted_files)
            for f in all_files:
                norm_f = os.path.normpath(f)
                if norm_f not in file_to_agents:
                    file_to_agents[norm_f] = set()
                file_to_agents[norm_f].add(agent_id)

        # Second pass: Identify cross-worktree file overlaps
        for norm_f, agents in file_to_agents.items():
            if len(agents) > 1:
                agent_list = sorted(list(agents))
                for agent_id in agent_list:
                    res = results[agent_id]
                    res.overlapping_files.append(norm_f)
                    other_agents = [a for a in agent_list if a != agent_id]
                    res.leaks.append(
                        IsolationLeak(
                            agent_id=agent_id,
                            leak_type=LeakType.CROSS_WORKTREE_OVERLAP,
                            filepath=norm_f,
                            reason=f"File '{norm_f}' was modified concurrently by agents: {', '.join(agent_list)}.",
                            details={"conflicting_agents": other_agents}
                        )
                    )

        return results

    def validate_pre_merge(
        self,
        worktree_path: str,
        agent_id: str = "agent",
        allowed_paths: Optional[List[str]] = None
    ) -> WorktreeDiffResult:
        """
        Validates a worktree prior to merging.

        Args:
            worktree_path (str): Worktree directory path.
            agent_id (str): Subagent ID.
            allowed_paths (Optional[List[str]]): Allowed file paths for the agent.

        Returns:
            WorktreeDiffResult: Result if clean.

        Raises:
            WorktreeContaminationError: If any isolation leaks or contamination are detected.
        """
        result = self.evaluate_worktree(worktree_path=worktree_path, agent_id=agent_id, allowed_paths=allowed_paths)
        if result.is_contaminated:
            leak_summaries = "; ".join(f"[{leak.leak_type}] {leak.filepath}: {leak.reason}" for leak in result.leaks)
            raise WorktreeContaminationError(
                f"Worktree merge validation failed for agent '{agent_id}'. Leaks detected: {leak_summaries}"
            )
        return result

    def _is_protected(self, filepath: str) -> bool:
        """Checks if filepath matches any protected patterns."""
        norm_p = os.path.normpath(filepath)
        for pattern in self.protected_patterns:
            if fnmatch.fnmatch(norm_p, pattern) or fnmatch.fnmatch(filepath, pattern):
                return True
            if "/" in pattern or os.sep in pattern:
                if fnmatch.fnmatch(filepath, pattern):
                    return True
            # Also check dirname or filename against pattern
            parts = filepath.split(os.sep)
            if any(fnmatch.fnmatch(part, pattern.rstrip("/*")) for part in parts):
                return True
        return False

    def _is_allowed(self, filepath: str, allowed_paths: List[str]) -> bool:
        """Checks if filepath matches any allowed path patterns."""
        norm_p = os.path.normpath(filepath)
        for pattern in allowed_paths:
            norm_pattern = os.path.normpath(pattern)
            if fnmatch.fnmatch(norm_p, norm_pattern) or fnmatch.fnmatch(filepath, pattern):
                return True
            # Check if filepath is inside allowed folder
            if filepath.startswith(pattern.rstrip("/*") + os.sep) or norm_p.startswith(norm_pattern.rstrip("/*") + os.sep):
                return True
            if norm_p == norm_pattern:
                return True
        return False
