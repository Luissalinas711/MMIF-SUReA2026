# Scores the main grid and writes metrics_long.csv
# 104 pairs x (15 configurations + 2 unfused references) x 7 metrics = 12376 rows.
# The scoring lives in src/evaluation.py so other scripts can use it too (originally ran in here only)

from evaluation import score_everything


# Quick look so a broken combination shows up now rather than later in the statistics script
def print_means(rows, metric='edge_preservation'):
    by_combination = {}

    for row in rows:
        if row['metric'] == metric:
            by_combination.setdefault((row['transform'], row['rule']), []).append(row['value'])

    print(f'\n  mean {metric}')

    for (transform, rule), values in sorted(by_combination.items()):
        label = f'{transform} + {rule}'
        print(f'    {label:<26} {sum(values) / len(values):.4f}')


if __name__ == '__main__':
    RESULTS = f'{DRIVE_ROOT}/results_grid'
    OUTPUT = f'{DRIVE_ROOT}/metrics_long.csv'

    rows = score_everything(DATA, RESULTS, OUTPUT)
    print_means(rows)
