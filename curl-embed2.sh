#!/bin/sh

curl -X POST http://localhost:7999/embed \
-H "Content-Type: application/json" \
-d '{
"texts": [ "吾輩は猫である。" ]
}'
