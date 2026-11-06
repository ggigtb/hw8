# -*- coding: utf-8 -*-
"""作业八拓展实验 C（自蒸馏）与 E（教师规模扫描）。"""
import json
import math
import os
import sys

import torch

import distill_lm as D

OUT = D.OUT_DIR
torch.set_num_threads(8)


def build(cfg):
    return D.GPT(V, cfg)


def distill_one(teacher, train_ids, val_ids, T=4.0, alpha=0.5, iters=120, seed=0):
    torch.manual_seed(seed + 7)
    s = D.GPT(V, D.STUDENT_CFG)
    _, tcost = D.train(s, train_ids, iters, BS, BLOCK, lr=3e-3, seed=seed,
                       teacher=teacher, T=T, alpha=alpha, log_every=100, tag="kd")
    v = D.evaluate(s, val_ids, BS, BLOCK)
    return v, math.exp(v), D.agreement(s, teacher, val_ids, BS, BLOCK), tcost


def main():
    global V, BS, BLOCK
    tok, train_ids, val_ids = D.make_data(D.TEACHER_CFG["block_size"])
    V = tok.vocab_size
    BS, BLOCK = 24, 96
    out = {}

    # ---------- C. 自蒸馏：教师 = 同结构但训练更久的学生 ----------
    torch.manual_seed(0 + 7)
    big_student = D.GPT(V, D.STUDENT_CFG)
    _, tcost = D.train(big_student, train_ids, 400, BS, BLOCK, lr=3e-3, seed=0,
                       log_every=200, tag="self-teacher")
    tv = D.evaluate(big_student, val_ids, BS, BLOCK)
    sv, sppl, sagree, _ = distill_one(big_student, train_ids, val_ids)
    out["C_self_distill"] = {
        "long_student_params": big_student.n_param,
        "long_student_val": tv, "long_student_ppl": math.exp(tv),
        "long_student_time_s": tcost,
        "distilled_val": sv, "distilled_ppl": sppl,
        "agree_with_long_student": sagree,
    }
    print("[C] long-student val=%.4f  ->  distilled val=%.4f" % (tv, sv), flush=True)

    # ---------- E. 教师规模扫描 ----------
    scan = []
    for C in (96, 160, 256):
        cfg = dict(n_layer=2, n_head=4, n_embd=C, block_size=96, dropout=0.0)
        torch.manual_seed(0)
        teacher = D.GPT(V, cfg)
        _, ttime = D.train(teacher, train_ids, 200, BS, BLOCK, lr=3e-3, seed=0,
                           log_every=200, tag="teacher-C%d" % C)
        tv = D.evaluate(teacher, val_ids, BS, BLOCK)
        kv, kppl, kagree, _ = distill_one(teacher, train_ids, val_ids)
        row = {"n_embd": C, "params": teacher.n_param, "teacher_val": tv,
               "teacher_ppl": math.exp(tv), "teacher_time_s": ttime,
               "kd_val": kv, "kd_ppl": kppl, "kd_agree": kagree}
        scan.append(row)
        print("[E] C=%d params=%.3fM teacher_val=%.4f kd_val=%.4f"
              % (C, teacher.n_param / 1e6, tv, kv), flush=True)
    out["E_teacher_scan"] = scan

    with open(os.path.join(OUT, "results_extras.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[done] %s" % os.path.join(OUT, "results_extras.json"), flush=True)


if __name__ == "__main__":
    main()
