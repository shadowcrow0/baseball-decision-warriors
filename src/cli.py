"""分析程式共用的命令列讀法：資料夾、球季、輸出資料夾，加上升級與 WAR 模型的 tag。

    args = read_args("這支程式在做什麼")
    args.data_dir, args.season, args.out_dir, args.promoted, args.war, args.score

每支分析程式的開頭都一樣，所以放在一起；fit_year.py 自己有更多選項，不用這個。
"""

import argparse


def read_args(description: str, need_war: bool = True, score: bool = False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("data_dir", help="資料夾（含 model_bat.parquet）")
    p.add_argument("season", type=int, help="球季")
    p.add_argument("out_dir", nargs="?", default="outputs", help="輸出資料夾（預設 outputs）")
    p.add_argument("--promoted", required=True, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--war", required=need_war, default=None, help="WAR 模型的 tag，例如 war_f4sed_linear_p2")
    if score:
        p.add_argument(
            "--score", default="exp_war", choices=["exp_war", "score_promoted"], help="「模型看好」用哪個分數排序"
        )
    return p.parse_args()
