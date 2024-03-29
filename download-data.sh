DATA_DIR="data"
test -d "$DATA_DIR" || mkdir "$DATA_DIR"
test -f "$DATA_DIR/test_for_report-stories_labels.csv" || curl "http://n.ethz.ch/~thomasdi/download/test_for_report-stories_labels.csv" --output "$DATA_DIR/test_for_report-stories_labels.csv"
test -f "$DATA_DIR/test-stories.csv" || curl "http://n.ethz.ch/~thomasdi/download/test-stories.csv" --output "$DATA_DIR/test-stories.csv"
test -f "$DATA_DIR/sct_train.csv" || curl "http://n.ethz.ch/~thomasdi/download/sct_train.csv" --output "$DATA_DIR/sct_train.csv"
test -f "$DATA_DIR/sct_val.csv" || curl "http://n.ethz.ch/~thomasdi/download/sct_val.csv" --output "$DATA_DIR/sct_val.csv"
test -f "$DATA_DIR/glove.6B.100d.txt" || curl "http://n.ethz.ch/~thomasdi/download/glove.6B.100d.txt" --output "$DATA_DIR/glove.6B.100d.txt"
