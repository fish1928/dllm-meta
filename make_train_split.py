#################################################
# Materialize the router-TRAINING subset of an oracle collection under the
# head-split scheme (full-benchmark p100 mockups, head collection):
#
#   docs 0..N-1 were collected in sample folders 0..N-1 (head of the task);
#   the LAST tail fraction of those folders becomes training data, the first
#   (1 - tail) stay reserved for end-to-end evaluation (run e2e with
#   LIMIT = (1 - tail) * baseline LIMIT, e.g. 500 -> 450).
#
# Creates <folder_dst> with symlinks to the last K complete sample folders of
# <folder_src>, renumbered 0..K-1 (RouterTrainer lists digit folders, so the
# destination folder exposes ONLY training samples -- pointing the trainer at
# the source folder would leak the e2e docs into training). A split_meta.json
# records the mapping for auditability.
#
# Usage:
#   python make_train_split.py --folder_src stats_oracle/llada_base_gsm8k_b1 \
#       --folder_dst stats_train/llada_base_gsm8k_b1 [--tail_percent 0.1 | --tail_count 50]
#
#   for d in stats_oracle/llada_base_*_b1; do
#       python make_train_split.py --folder_src "$d" --folder_dst "stats_train/$(basename $d)"
#   done
#################################################

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_src', type=str, required=True)
    parser.add_argument('--folder_dst', type=str, required=True)
    parser.add_argument('--tail_percent', type=float, default=0.1,
                        help='fraction of the collection used for training (default 0.1)')
    parser.add_argument('--tail_count', type=int, default=None,
                        help='absolute count instead of a fraction (e.g. 50)')
    return parser.parse_args()
# end


def main():
    args = parse_args()

    ids_complete = sorted(
        int(name) for name in os.listdir(args.folder_src)
        if name.isdigit() and os.path.exists(os.path.join(args.folder_src, name, 'generated.json'))
    )
    assert ids_complete, f'no complete sample folders in {args.folder_src}'

    n_tail = args.tail_count if args.tail_count is not None \
        else max(1, int(len(ids_complete) * args.tail_percent))
    ids_train = ids_complete[-n_tail:]

    os.makedirs(args.folder_dst, exist_ok=True)
    for id_new, id_src in enumerate(ids_train):
        path_link = os.path.join(args.folder_dst, str(id_new))
        if os.path.islink(path_link):
            os.remove(path_link)
        # end
        os.symlink(
            os.path.relpath(os.path.join(args.folder_src, str(id_src)), args.folder_dst),
            path_link,
        )
    # end

    with open(os.path.join(args.folder_dst, 'split_meta.json'), 'w') as file:
        json.dump({
            'folder_src': args.folder_src,
            'n_complete_src': len(ids_complete),
            'ids_train_src': ids_train,
            'n_train': len(ids_train),
            'ids_reserved_for_e2e': f'0..{ids_train[0] - 1} (run e2e with a LIMIT that stays below {ids_train[0]} per task)',
        }, file, indent=2)
    # end

    print(f'{args.folder_dst}: {len(ids_train)} train samples '
          f'(src ids {ids_train[0]}..{ids_train[-1]} of {len(ids_complete)} complete); '
          f'e2e-safe LIMIT for this task: {ids_train[0]}')
# end


if __name__ == '__main__':
    main()
# end
