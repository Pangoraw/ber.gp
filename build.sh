#!/bin/bash
pushd themes/organic
npm i -g postcss@7.0.35 postcss-cli@8.3.0 autoprefixer@10.0.1 tailwindcss@compat
npm i
popd

hugo --gc --minify

ls -lR public/css

