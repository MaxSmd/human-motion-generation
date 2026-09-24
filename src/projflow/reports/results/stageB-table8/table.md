| Joint | cells | fid | fid_processed | fid_floor | r3 | diversity | mm_dist | skate_upstream | skate_ours | traj_err | loc_err | avg_err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Pelvis** | 5 | 0.1803 | 0.0977 | 0.0642 | 0.7869 | 9.5780 | 3.0643 | 0.0663 | 0.4191 | 0.0000 | 0.0000 | 0.0000 |
| Pelvis (paper T8) | | 0.0830 |  |  | 0.7550 | 9.0960 |  | 0.0651 |  |  |  |  |
| **Left foot** | 5 | 0.1978 | 0.1119 | 0.0638 | 0.7683 | 9.6433 | 3.1915 | 0.0632 | 0.3876 | 0.0000 | 0.0000 | 0.0000 |
| **Right foot** | 5 | 0.1994 | 0.1171 | 0.0641 | 0.7737 | 9.5855 | 3.1991 | 0.0639 | 0.3866 | 0.0000 | 0.0000 | 0.0000 |
| **Head** | 5 | 0.2203 | 0.0990 | 0.0638 | 0.7847 | 9.7270 | 3.1404 | 0.0626 | 0.3668 | 0.0000 | 0.0000 | 0.0000 |
| **Left wrist** | 5 | 0.1595 | 0.0808 | 0.0642 | 0.7850 | 9.4227 | 3.1097 | 0.0617 | 0.3776 | 0.0000 | 0.0000 | 0.0000 |
| **Right wrist** | 5 | 0.1623 | 0.0825 | 0.0640 | 0.7850 | 9.4855 | 3.1075 | 0.0621 | 0.3763 | 0.0000 | 0.0000 | 0.0000 |
| **Average** | 30 | 0.1866 | 0.0982 | 0.0640 | 0.7806 | 9.5737 | 3.1354 | 0.0633 | 0.3857 | 0.0000 | 0.0000 | 0.0000 |
| Average (paper T8) | | 0.0740 |  |  | 0.7520 | 9.0650 |  | 0.0624 |  |  |  |  |

Real-motion reference (mean over cells) — canonical: R@3 0.7829, Diversity 9.2680; processed: R@3 0.7834, Diversity 9.4770; paper legacy GT: R@3 0.797, Diversity 9.503. `fid` scores against canonical real features, `fid_processed` against real joints through the same process_file path, `fid_floor` is canonical vs processed real.
