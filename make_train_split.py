#################################################
# Materialize the router-TRAINING subset of oracle collections under the
# head-split scheme (full-benchmark p100 mockups, head collection):
#
#   docs 0..N-1 were collected in sample folders 0..N-1 (head of the task);
#   the LAST tail fraction of those folders becomes training data, the first
#   (1 - tail) stay reserved for end-to-end evaluation (run e2e with
#   LIMIT = (1 - tail) * baseline LIMIT, e.g. 500 -> 450).
#
# For each collection it creates a destination folder holding REAL COPIES of
# the last K complete sample folders of the source, renumbered 0..K-1
# (RouterTrainer lists digit folders, so the destination exposes ONLY
# training samples -- pointing the trainer at the source folder would leak
# the e2e docs into training). A split_meta.json records the mapping.
#
# RESUME: a rerun keeps destination sample folders that already contain
# generated.json (finished copies) and only copies what is missing; partial
# copies from an interrupted run and symlinks from the old scheme are
# replaced by fresh full copies.
#
# PARALLEL: collections run concurrently (--workers), and each collection's
# completeness scan fans its generated.json stat calls out over its own
# thread pool (--stat_workers) -- both matter on a big/network filesystem.
# The tail copies themselves stream at --workers concurrent collections.
#
# BATCH mode (all collections under one root, e.g. every thread/task/block):
#   python make_train_split.py --root stats_oracle [--out stats_train] \
#       [--tail_percent 0.1 | --tail_count 50] [--workers 8] [--stat_workers 32]
#   Every direct subfolder of --root containing digit sample folders is
#   split into <out>/<same basename>. Collections with no complete samples
#   are reported and skipped; folders without digit subfolders are ignored.
#
# SINGLE-collection mode (unchanged):
#   python make_train_split.py --folder_src stats_oracle/llada_base_gsm8k_b1 \
#       --folder_dst stats_train/llada_base_gsm8k_b1 [--tail_count 50]
#################################################

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None,
                        help='batch mode: split EVERY collection folder under this root')
    parser.add_argument('--out', type=str, default='stats_train',
                        help='batch mode: destination root (default stats_train)')
    parser.add_argument('--folder_src', type=str, default=None)
    parser.add_argument('--folder_dst', type=str, default=None)
    parser.add_argument('--tail_percent', type=float, default=0.1,
                        help='fraction of the collection used for training (default 0.1)')
    parser.add_argument('--tail_count', type=int, default=None,
                        help='absolute count instead of a fraction (e.g. 50)')
    parser.add_argument('--workers', type=int, default=8,
                        help='collections processed concurrently (default 8)')
    parser.add_argument('--stat_workers', type=int, default=32,
                        help='parallel completeness stats per collection (default 32)')
    args = parser.parse_args()

    if args.root is None and (args.folder_src is None or args.folder_dst is None):
        parser.error('pass --root <stats_oracle> for batch mode, '
                     'or both --folder_src and --folder_dst for a single collection')
    # end
    return args
# end


def scan_complete_ids(folder_src, stat_workers):
    """Sorted ids of sample folders whose generated.json exists; the stat
    calls are the hot path on network filesystems, so fan them out."""
    names_digit = [name for name in os.listdir(folder_src) if name.isdigit()]
    if not names_digit:
        return None    # not a collection folder at all

    def is_complete(name):
        return os.path.exists(os.path.join(folder_src, name, 'generated.json'))
    # end

    if stat_workers > 1 and len(names_digit) > 16:
        with ThreadPoolExecutor(max_workers=stat_workers) as pool:
            flags = list(pool.map(is_complete, names_digit))
    else:
        flags = [is_complete(name) for name in names_digit]
    # end
    return sorted(int(name) for name, ok in zip(names_digit, flags) if ok)
# end


def split_one(folder_src, folder_dst, tail_percent, tail_count, stat_workers=32):
    """Symlink the tail of one collection into folder_dst. Returns a summary
    dict; {'empty': True} when nothing is complete; None when folder_src has
    no digit subfolders (not a collection)."""
    ids_complete = scan_complete_ids(folder_src, stat_workers)
    if ids_complete is None:
        return None
    if not ids_complete:
        return {'empty': True}
    # end

    n_tail = tail_count if tail_count is not None \
        else max(1, int(len(ids_complete) * tail_percent))
    ids_train = ids_complete[-n_tail:]

    os.makedirs(folder_dst, exist_ok=True)
    num_copied, num_kept = 0, 0
    for id_new, id_src in enumerate(ids_train):
        path_dst = os.path.join(folder_dst, str(id_new))

        # resume: a REAL directory that already has generated.json is a
        # finished copy of this slot -- keep it (a rerun must not recopy 10G).
        # Anything else (symlink from the old scheme, partial copy from an
        # interrupted run) is replaced by a fresh full copy.
        if os.path.islink(path_dst):
            os.remove(path_dst)
        elif os.path.isdir(path_dst):
            if os.path.exists(os.path.join(path_dst, 'generated.json')):
                num_kept += 1
                continue
            shutil.rmtree(path_dst)
        # end

        shutil.copytree(os.path.join(folder_src, str(id_src)), path_dst)
        num_copied += 1
    # end

    with open(os.path.join(folder_dst, 'split_meta.json'), 'w') as file:
        json.dump({
            'folder_src': folder_src,
            'mode': 'copy',
            'n_complete_src': len(ids_complete),
            'ids_train_src': ids_train,
            'n_train': len(ids_train),
            'ids_reserved_for_e2e': f'0..{ids_train[0] - 1} (run e2e with a LIMIT that stays below {ids_train[0]} per task)',
        }, file, indent=2)
    # end

    return {'n_complete': len(ids_complete), 'ids_train': ids_train,
            'limit_e2e_safe': ids_train[0],
            'num_copied': num_copied, 'num_kept': num_kept}
# end


def report_line(folder_src, folder_dst, summary):
    if summary is None:
        return None    # not a collection: stay quiet
    if summary.get('empty'):
        return f'SKIP {folder_src}: no complete sample folders (generated.json missing)'
    return (f'{folder_dst}: {len(summary["ids_train"])} train samples '
            f'({summary["num_copied"]} copied, {summary["num_kept"]} already present; '
            f'src ids {summary["ids_train"][0]}..{summary["ids_train"][-1]} '
            f'of {summary["n_complete"]} complete); '
            f'e2e-safe LIMIT for this task: {summary["limit_e2e_safe"]}')
# end


def main():
    args = parse_args()

    if args.root is None:
        pairs = [(args.folder_src, args.folder_dst)]
    else:
        pairs = [(os.path.join(args.root, name), os.path.join(args.out, name))
                 for name in sorted(os.listdir(args.root))
                 if os.path.isdir(os.path.join(args.root, name))]
        assert pairs, f'no subfolders under {args.root}'
    # end

    num_done, num_empty, num_failed = 0, 0, 0
    limits = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(split_one, src, dst, args.tail_percent, args.tail_count,
                        args.stat_workers): (src, dst)
            for src, dst in pairs
        }
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                summary = future.result()
            except Exception as error:
                print(f'FAILED {src}: {error}')
                num_failed += 1
                continue
            # end

            line = report_line(src, dst, summary)
            if line:
                print(line, flush=True)
            # end
            if summary is None:
                continue
            if summary.get('empty'):
                num_empty += 1
            else:
                num_done += 1
                limits[os.path.basename(dst)] = summary['limit_e2e_safe']
            # end
        # end
    # end

    if args.root is not None:
        print(f'\nbatch done: {num_done} collections split, {num_empty} skipped, '
              f'{num_failed} failed -> {args.out}')
        if limits:
            limit_min = min(limits.values())
            print(f'e2e-safe LIMIT across all splits: {limit_min} '
                  f'(tightest: {", ".join(sorted(n for n, v in limits.items() if v == limit_min))})')
        # end
    # end
# end


if __name__ == '__main__':
    main()
# end
