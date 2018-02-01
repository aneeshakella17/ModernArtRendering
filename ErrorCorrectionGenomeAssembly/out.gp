set terminal png tiny size 800,800
set output "out.png"
set size 1,1
set grid
unset key
set border 15
set tics scale 0
set xlabel "chr3"
set ylabel "chr3"
set format "%.0f"
if(GPVAL_VERSION < 5) { set mouse clipboardformat "[%.0f, %.0f]" }
set xrange [1:1230248]
set yrange [1:1297030]
set style line 1  lt 1 lw 3 pt 6 ps 1
set style line 2  lt 3 lw 3 pt 6 ps 1
set style line 3  lt 2 lw 3 pt 6 ps 1
plot \
 "out.fplot" title "FWD" w lp ls 1, \
 "out.rplot" title "REV" w lp ls 2
