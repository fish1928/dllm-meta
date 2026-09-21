#################################################
# Materialize the router-TRAINING subset of oracle collections under the
# head-split scheme (full-benchmark p100 mockups, head collection):
#
#   docs 0..N-1 were collected in sample folders 0..N-1 (head of the task);
#   the LAST tail fraction of those folders becomes training data, the first
#   (1 - tail) stay reserved for end-to-end evaluation (run e2e with
#   LIMIT = (1 - tail) * baseline LIMIT, e.g. 500 -> 450).
#
# For each collection it creates a destination folder with symlinks to the
# last K complete sample folders of the source, renumbered 0..K-1
# (RouterTrainer lists digit folders, so the destination exposes ONLY
# training samples -- pointing the trainer at the source folder would leak
# the e2e docs into training). A split_meta.json records the mapping.
#
# BATCH mode (all collections under one root, e.g. every thread/task/block):
#   python make_train_split.py --root stats_oracle [--out stats_train] \
#       [--tail_percent 0.1 | --tail_count 50]
#   Every direct subfolder of --root that contains digit sample folders is
#   split into <out>/<same basename>. Collections with no complete samples
#   are reported and skipped, not fatal.
#
# SINGLE-collection mode (unchanged):
#   python make_train_split.py --folder_src stats_oracle/llada_base_gsm8k_b1 \
#       --folder_dst stats_train/llada_base_gsm8k_b1 [--tail_count 50]
#################################################

import argparse
import json
import os


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
    args = parser.parse_args()

    if args.root is None and (args.folder_src is None or args.folder_dst is None):
        parser.error('pass --root <stats_oracle> for batch mode, '
                     'or both --folder_src and --folder_dst for a single collection')
    # end
    return args
# end


def split_one(folder_src, folder_dst, tail_percent, tail_count):
    """Symlink the tail of one collection into folder_dst; returns a summary
    dict, or None when the source has no complete sample folders."""
    ids_complete = sorted(
        int(name) for name in os.listdir(folder_src)
        if name.isdigit() and os.path.exists(os.path.join(folder_src, name, 'generated.json'))
    )
    if not ids_complete:
        return None
    # end

    n_tail = tail_count if tail_count is not None \
        else max(1, int(len(ids_complete) * tail_percent))
    ids_train = ids_complete[-n_tail:]

    os.makedirs(folder_dst, exist_ok=True)
    for id_new, id_src in enumerate(ids_train):
        path_link = os.path.join(folder_dst, str(id_new))
        if os.path.islink(path_link):
            os.remove(path_link)
        # end
        os.symlink(
            os.path.relpath(os.path.join(folder_src, str(id_src)), folder_dst),
            path_link,
        )
    # end

    with open(os.path.join(folder_dst, 'split_meta.json'), 'w') as file:
        json.dump({
            'folder_src': folder_src,
            'n_complete_src': len(ids_complete),
            'ids_train_src': ids_train,
            'n_train': len(ids_train),
            'ids_reserved_for_e2e': f'0..{ids_train[0] - 1} (run e2e with a LIMIT that stays below {ids_train[0]} per task)',
        }, file, indent=2)
    # end

    return {'n_complete': len(ids_complete), 'ids_train': ids_train,
            'limit_e2e_safe': ids_train[0]}
# end


def main():
    args = parse_args()

    if args.root is None:
        pairs = [(args.folder_src, args.folder_dst)]
    else:
        names = sorted(
            name for name in os.listdir(args.root)
            if os.path.isdir(os.path.join(args.root, name))
            and any(child.isdigit()
                    for child in os.listdir(os.path.join(args.root, name)))
        )
        assert names, f'no collection folders (with digit sample subfolders) under {args.root}'
        pairs = [(os.path.join(args.root, name), os.path.join(args.out, name))
                 for name in names]
    # end

    num_done, num_empty = 0, 0
    limits = {}
    for folder_src, folder_dst in pairs:
        summary = split_one(folder_src, folder_dst, args.tail_percent, args.tail_count)
        if summary is None:
            print(f'SKIP {folder_src}: no complete sample folders (generated.json missing)')
            num_empty += 1
            continue
        # end
        num_done += 1
        limits[os.path.basename(folder_dst)] = summary['limit_e2e_safe']
        print(f'{folder_dst}: {len(summary["ids_train"])} train samples '
              f'(src ids {summary["ids_train"][0]}..{summary["ids_train"][-1]} '
              f'of {summary["n_complete"]} complete); '
              f'e2e-safe LIMIT for this task: {summary["limit_e2e_safe"]}')
    # end

    if args.root is not None:
        print(f'\nbatch done: {num_done} collections split, {num_empty} skipped -> {args.out}')
        if limits:
            limit_min = min(limits.values())
            print(f'e2e-safe LIMIT across all splits: {limit_min} '
                  f'(tightest: {", ".join(n for n, v in limits.items() if v == limit_min)})')
        # end
    # end
# end


if __name__ == '__main__':
    main()
# end
