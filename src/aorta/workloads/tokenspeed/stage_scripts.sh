#!/usr/bin/env bash
# Copy the TokenSpeed probe scripts to a node-local staging directory.
#
# Needed because the docker daemon runs as root, so on a root-squashed NFS home
# `-v /home/<user>/...` fails with "permission denied" and the scripts have to
# live on the compute node's own disk. Editing the source tree and forgetting to
# re-copy is the easiest way to waste a matrix run, so this always mirrors the
# full set rather than copying files one at a time.
#
# Run this ON the compute node. /tmp is per-node, so staging from a login node
# writes a directory the trials never see -- and if the compute node already has
# an older copy, the run silently succeeds against stale scripts.
#
# The .py helpers are staged too, so that list_harness_coverage.py (which runs
# inside the container) can be bind-mounted from here without a second copy.
#
# Usage:
#   bash stage_scripts.sh [dest]     # dest defaults to /tmp/ts-work/scripts

set -euo pipefail

dest="${1:-/tmp/ts-work/scripts}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Staging into the source directory would be destructive, not merely pointless:
# the mirror step below deletes the set this script owns before copying, so with
# dest == src it deletes every probe script in the tree and then fails with
# nothing left to copy. `dest` is a positional argument, and "stage the scripts
# where the scripts are" is an easy thing to type.
#
# Checked before the NFS guard, which would otherwise mask it whenever the tree
# happens to live under /home -- with a message pointing at the wrong problem,
# and no coverage of this guard at all on a checkout that does not.
#
# Compared after resolving both, so a relative path, a trailing slash or a
# symlink into the tree cannot walk around the check.
src_real="$(cd "${src}" && pwd -P)"
if [ -d "${dest}" ] && [ "$(cd "${dest}" && pwd -P)" = "${src_real}" ]; then
  echo "stage_scripts: dest is the source directory (${src_real})." >&2
  echo "  Staging there would delete the scripts it is meant to copy." >&2
  echo "  Pick a node-local path such as /tmp/ts-work/scripts." >&2
  exit 64
fi

# What actually matters is the filesystem type, not how the path is spelled.
# The old `/home/*|/nfs/*` test was wrong in both directions: it passed `/mnt`,
# `/shared`, `/users` and any autofs path -- which is where a cluster usually
# puts its network storage -- while rejecting a perfectly local `/home` on an
# ordinary workstation. The mount is resolved from /proc/mounts by longest
# matching prefix, which is the mount the kernel would use.
#
# Best-effort by construction: no /proc/mounts, or a path whose mount cannot be
# identified, is treated as local rather than blocking staging on a guess.
# harvest_code_objects.py carries the same logic as `_network_filesystem`.
_NETWORK_FSTYPES=" afs beegfs ceph cifs fuse.cephfs fuse.glusterfs fuse.sshfs gfs2 glusterfs gpfs lustre nfs nfs4 smb3 "
network_fstype() {  # network_fstype <path>
  local target="$1" best_len=-1 best_fs="" dev mount fstype rest dir base
  local mounts="${TS_MOUNTS_FILE:-/proc/mounts}"
  [ -r "${mounts}" ] || return 0
  dir="$(dirname "${target}")"
  base="$(basename "${target}")"
  dir="$(cd "${dir}" 2>/dev/null && pwd -P)" || return 0
  target="${dir%/}/${base}"
  while read -r dev mount fstype rest; do
    # /proc/mounts octal-escapes spaces and tabs in the mount point.
    mount="${mount//\\040/ }"
    mount="${mount//\\011/	}"
    mount="${mount%/}"
    case "${target}/" in
      "${mount}"/*) ;;
      *) continue ;;
    esac
    if [ "${#mount}" -gt "${best_len}" ]; then
      best_len="${#mount}"
      best_fs="${fstype}"
    fi
  done < "${mounts}"
  case "${_NETWORK_FSTYPES}" in
    *" ${best_fs} "*) printf '%s' "${best_fs}" ;;
  esac
}

_dest_fstype="$(network_fstype "${dest}")"
if [ -n "${_dest_fstype}" ]; then
  echo "stage_scripts: ${dest} is on ${_dest_fstype}; the docker daemon cannot read it." >&2
  echo "  Pick a node-local path such as /tmp/ts-work/scripts." >&2
  exit 64
fi

mkdir -p "${dest}"

# Re-checked now that the directory exists: `mkdir -p` can have created a path
# that resolves into the tree (a symlink component, say) which the pre-flight
# check above could not resolve because it was not there yet.
if [ "$(cd "${dest}" && pwd -P)" = "${src_real}" ]; then
  echo "stage_scripts: dest resolves to the source directory (${src_real})." >&2
  echo "  Staging there would delete the scripts it is meant to copy." >&2
  exit 64
fi

# Clear the set this script owns before copying, so the destination really is a
# mirror. A plain `cp` only ever adds: a probe that was renamed or deleted
# upstream stays behind in the staging directory, and because the recipes name
# their entry script by filename, a stale copy is not just dead weight -- it is
# still executable, and a run pointed at the old name succeeds against code that
# no longer exists in the tree.
#
# What gets removed is read from a manifest this script wrote on its last run,
# not from a `*.sh` / `*.py` glob. `dest` is a positional argument, so the glob
# deleted whatever happened to match in a directory this script does not own:
# `stage_scripts.sh /tmp` removed every other user's staging scripts before
# copying. The manifest keeps the mirror property while limiting deletion to
# files this script actually put there.
#
# First run into an existing directory therefore deletes nothing, which is the
# right default: with no manifest there is no evidence any of those files came
# from here.
manifest="${dest}/.aorta-staged"

# Not through a symlink: `-f` follows one, and in a shared directory a planted
# link would make the deletion list somebody else's to write. A run written by
# this script is always a regular file, so refusing the link loses nothing.
if [ -L "${manifest}" ]; then
  echo "stage_scripts: ${manifest} is a symlink; refusing to trust it." >&2
  echo "  Remove it, or stage into a directory only you can write." >&2
  exit 64
fi

# The manifest is a *deletion authority*, so it has to be ours. Refusing the
# symlink closed one way to supply that list; a co-tenant can equally write a
# plain `.aorta-staged` naming `cell.env`, and this loop would then delete a
# file this script never staged. The basename check only stops traversal, not
# that. Owned by this uid, or it is not trusted -- which for a first run into a
# populated directory means deleting nothing, the same safe default as no
# manifest at all.
_manifest_owner=""
if [ -f "${manifest}" ]; then
  _manifest_owner="$(stat -c '%u' "${manifest}" 2>/dev/null || echo "")"
fi
if [ -f "${manifest}" ] && [ "${_manifest_owner}" != "$(id -u)" ]; then
  echo "stage_scripts: ${manifest} is not owned by this user; ignoring it." >&2
  echo "  Nothing will be removed from ${dest} on this run. Stage into a" >&2
  echo "  directory only you can write to restore mirror behaviour." >&2
elif [ -f "${manifest}" ]; then
  while IFS= read -r staged; do
    # Basenames only, and re-checked here: a hand-edited manifest must not be
    # able to reach outside the staging directory.
    case "${staged}" in
      ''|*/*|.|..) continue ;;
    esac
    # Never through a link, for the same reason as the manifest itself.
    [ -L "${dest}/${staged}" ] && continue
    rm -f "${dest:?}/${staged}"
  done < "${manifest}"
fi
rm -rf "${dest}/__pycache__"

# Each file lands via a private temporary and a rename, for the same reason the
# manifest does: `cp` follows a symlink at the destination, so in a shared `dest`
# another user could plant `host_launch.sh -> <something the caller can write>`
# and have staging overwrite that target instead. The rename replaces the link
# rather than following it, and is atomic, so a concurrent run never executes a
# half-copied script.
for f in "${src}"/*.sh "${src}"/*.py; do
  base="$(basename "${f}")"
  tmp="$(mktemp "${dest}/.aorta-stage.XXXXXX")" || {
    echo "stage_scripts: cannot create a temporary file in ${dest}" >&2
    exit 64
  }
  if ! cat "${f}" > "${tmp}"; then
    rm -f "${tmp}"
    echo "stage_scripts: failed to copy ${f}" >&2
    exit 64
  fi
  # 0755 for the shells and 0644 for the rest, set before the rename so the file
  # is never briefly visible under its real name with mktemp's 0600.
  case "${base}" in
    *.sh) chmod 755 "${tmp}" ;;
    *) chmod 644 "${tmp}" ;;
  esac
  mv -f "${tmp}" "${dest}/${base}"
done

# Written after the copy so an interrupted run cannot leave a manifest naming
# files that were never staged.
#
# Via a private temporary file and a rename, not by redirecting onto the path.
# `dest` is explicitly allowed to be a shared directory, and `>` follows a
# symlink: another user could pre-create `.aorta-staged` pointing at any file
# this caller can write and have this script truncate it. `mktemp` creates in
# the destination (so the rename stays on one filesystem) and refuses to follow
# anything, and the rename is atomic, so a concurrent staging run reads either
# the old manifest or the new one and never a half-written list.
manifest_tmp="$(mktemp "${dest}/.aorta-staged.XXXXXX")" || {
  echo "stage_scripts: cannot create a temporary manifest in ${dest}" >&2
  exit 64
}
trap 'rm -f "${manifest_tmp}"' EXIT
for f in "${src}"/*.sh "${src}"/*.py; do
  basename "${f}"
done > "${manifest_tmp}"
# 0644: the manifest is read by the next run, which may be another user's.
chmod 644 "${manifest_tmp}"
mv -f "${manifest_tmp}" "${manifest}"
trap - EXIT

echo "stage_scripts: staged to ${dest}"
# Iterating the manifest rather than globbing `dest`, for the same reason the
# removal above does: in a shared directory the glob would also pick up files
# this script never staged, and a syntax error in one of those would fail the
# staging run with exit 65.
while IFS= read -r staged; do
  f="${dest}/${staged}"
  # Syntax-check here rather than discovering a typo inside a container, where
  # the only symptom is a failed trial.
  case "${f}" in
    *.sh) checker=(bash -n "${f}") ;;
    *.py) checker=(python3 -m py_compile "${f}") ;;
  esac
  "${checker[@]}" || {
    echo "stage_scripts: syntax error in ${f}" >&2
    exit 65
  }
  printf '  %s (%s bytes)\n' "${staged}" "$(stat -c%s "${f}")"
done < "${manifest}"
# py_compile drops caches next to the source; they would otherwise be copied
# into the container on the next run for no reason.
rm -rf "${dest}/__pycache__"
echo "stage_scripts: syntax OK"
