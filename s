[33mcommit 2f5b41fecb55bf4608e0fe700acc7920a95b6e66[m[33m ([m[1;36mHEAD[m[33m, [m[1;31morigin/simplediff[m[33m)[m
Author: casperfibaek <casperfibaek@gmail.com>
Date:   Wed May 24 14:01:22 2023 +0200

    log

[1mdiff --git a/.gitignore b/.gitignore[m
[1mindex 722d5e7..7a53141 100644[m
[1m--- a/.gitignore[m
[1m+++ b/.gitignore[m
[36m@@ -1 +1,3 @@[m
 .vscode[m
[32m+[m[32mresults[m
[32m+[m[32mlogs[m
[1mdiff --git a/train.py b/train.py[m
[1mindex 73967ec..e6d49ee 100644[m
[1m--- a/train.py[m
[1m+++ b/train.py[m
[36m@@ -92,7 +92,8 @@[m [mdef main(args):[m
     cond, y_test = None, torch.full((args.n_samples,),1).to(device) if args.num_classes>0 else None[m
     dir = args.dir[m
     os.makedirs(dir,exist_ok=True)[m
[31m-    ckpt_best = os.path.join(dir, "best.pt") # do it into[m
[32m+[m[32m    os.makedirs(os.path.join(dir,"logs"),exist_ok=True)[m
[32m+[m[32m    ckpt_best = os.path.join(dir, "logs/best.pt") # do it into[m
     for i in range(args.epochs):[m
         model.train()[m
         for j,(data) in (enumerate(train_dataloader)):[m
