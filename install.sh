set -e

dir=$(dirname $0)

python3 -m venv "$dir/env"
source "$dir/env/bin/activate"
python3 -m pip install -r "$dir/devreqs.txt"
python3 -m pip install -e "$dir"
