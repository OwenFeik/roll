DIR=$(dirname $0)

source "$DIR/env/bin/activate" > /dev/null

echo -e "Black:\n" \
    && python -m black "$DIR/roll" "$DIR/tests" \
    && echo -e "\nMyPy:\n" \
    && python -m mypy "$DIR/roll" "$DIR/tests" \
    && echo -e "\nTests:\n" \
    && python "$DIR/tests/"*
