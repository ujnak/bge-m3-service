#!/bin/sh

curl -X POST http://localhost:7999/embed \
-H "Content-Type: application/json" \
-d '{
"texts": [ "これはテストです。", "This is a test." ]
}'
