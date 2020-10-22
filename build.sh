#!/bin/bash
pushd themes/organic
npm i -g postcss@8.1.0 postcss-cli@8.1.0 autoprefixer@10.0.1 tailwindcss@1.9.5
npm i
popd

hugo --gc --minify

ls -lR public/css

