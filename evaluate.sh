# python evaluateow.py -d sdfow -n 20240611bacow --train empty --test empty -g 1 &
# python evaluateow.py -d sdfow -n 20240611bac2ow --train regulator --test regulator -g 0 &
# python evaluateow.py -d sdfow -n 20240611bac2ow --train front_4250 --test front_4250 -g 5 &
# python evaluateow.py -d sdfow -n 20240611bac2ow --train tamaraw_2_6 --test tamaraw_2_6 -g 3 &
# python evaluateow.py -d sdfow -n 20240611bac2ow --train pred300_security_7_data_vs_time_20 --test pred300_security_7_data_vs_time_20 -g 5 &
# python evaluateow.py -d sdfow -n 20240611bac2ow --train gan300_tol6 --test gan300_tol6 -g 4 &
# wait
python evaluateow.py -d sdfow --attack DF -n 20240611bacow --train empty --test empty -g 1 &
python evaluateow.py -d sdfow --attack DF -n 20240611bacow --train regulator --test regulator -g 1 &
python evaluateow.py -d sdfow --attack DF -n 20240611bacow --train front_4250 --test front_4250 -g 0 &
python evaluateow.py -d sdfow --attack DF -n 20240611bacow --train tamaraw_2_6 --test tamaraw_2_6 -g 5 &
python evaluateow.py -d sdfow --attack DF -n 20240611bacow --train pred300_security_7_data_vs_time_20 --test pred300_security_7_data_vs_time_20 -g 5 &
python evaluateow.py -d sdfow --attack DF -n 20240611bacow --train gan300_tol6 --test gan300_tol6 -g 0 &
wait
