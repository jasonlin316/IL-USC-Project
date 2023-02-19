python products_gcn.py --algo mini
cd FlameGraph/
./flamegraph.pl --title "ogbn-products/mini-batch/gcn" --countname "us." /tmp/prod_mini_stacks.txt > prod_mini_stacks_gcn.svg
cd ..
python link_gcn.py --algo mini
cd FlameGraph/
./flamegraph.pl --title "ogbn-citation/mini-batch/gcn" --countname "us." /tmp/link_mini_stacks.txt > link_mini_stacks_gcn.svg
cd ..
python products_gcn.py --algo full
cd FlameGraph/
./flamegraph.pl --title "ogbn-products/full-graph/gcn" --countname "us." /tmp/prod_full_stacks.txt > prod_full_stacks_gcn.svg
cd ..
python link_gcn.py --algo full
cd FlameGraph/
./flamegraph.pl --title "ogbn-citation/full-graph/gcn" --countname "us." /tmp/link_full_stacks.txt > link_full_stacks_gcn.svg
cd ..