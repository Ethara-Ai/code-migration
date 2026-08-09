#!/bin/bash
# Grade with MigrationBench, which is already installed in the environment image
# along with JDK 17 and Maven -- so no network install here, and the JDK that
# built the classes is the JDK that inspects them.
set -o pipefail

mkdir -p /logs/verifier

python -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA 2>&1 \
  | tee /logs/verifier/pytest.log
status=${PIPESTATUS[0]}

if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit 0
