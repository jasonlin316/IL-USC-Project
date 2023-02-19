python products.py --algo mini
cd FlameGraph/
./flamegraph.pl --title "ogbn-products/mini-batch" --countname "us." /tmp/prod_mini_stacks.txt > prod_mini_stacks.svg
cd ..
python link.py --algo mini
cd FlameGraph/
./flamegraph.pl --title "ogbn-citation/mini-batch" --countname "us." /tmp/link_mini_stacks.txt > link_mini_stacks.svg
cd ..
python products.py --algo full
cd FlameGraph/
./flamegraph.pl --title "ogbn-products/full-graph" --countname "us." /tmp/prod_full_stacks.txt > prod_full_stacks.svg
cd ..
python link.py --algo full
cd FlameGraph/
./flamegraph.pl --title "ogbn-citation/full-graph" --countname "us." /tmp/link_full_stacks.txt > link_full_stacks.svg
cd ..