# Additional experiments summary

All statements below are generated from the experiment CSV files.

1. Adaptive fusion was not consistently better than fixed 0.5 across all evaluated datasets by mean seed-level PCC.
2. Mean selected alpha by dataset: CCLE=0.3100; ERKAUC30=0.2920; ERKIC50=0.2070; PI3KAUC=0.2070; PI3KIC50=0.2030; PRISM19Q4_independent=0.4500.
3. Mean PCC change from 10% to 30% masking: CCLE/mfmr_base=-0.0417; CCLE/original_mclrp=-0.0531; ERKAUC30/mfmr_base=-0.0710; ERKAUC30/original_mclrp=-0.0881; ERKIC50/mfmr_base=-0.0567; ERKIC50/original_mclrp=-0.0635; PI3KAUC/mfmr_base=-0.0494; PI3KAUC/original_mclrp=-0.0722; PI3KIC50/mfmr_base=-0.0536; PI3KIC50/original_mclrp=-0.0625.
4. MFMR had a smaller absolute PCC degradation slope than Original MCLRP in 5/5 comparable datasets.
5. Mean imputer/ridge prediction Pearson correlation: CCLE=0.9748; ERKAUC30=0.9324; ERKIC50=0.9532; PI3KAUC=0.9293; PI3KIC50=0.9277; PRISM19Q4_independent=0.9380.
6. Mean imputer/ridge error Pearson correlation: CCLE=0.9691; ERKAUC30=0.9296; ERKIC50=0.9583; PI3KAUC=0.9420; PI3KIC50=0.9409; PRISM19Q4_independent=0.9387.
7. Fixed fusion exceeded both individual branches in mean PCC in 5/6 datasets.
8. Fusion benefit relative to the better branch was larger in the highest than the lowest disagreement quartile.
