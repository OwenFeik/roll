DIR=$(dirname $0)

python -m black "$DIR/roll" "$DIR/tests"
python -m mypy "$DIR/roll" "$DIR/tests"
python "$DIR/tests/"*
