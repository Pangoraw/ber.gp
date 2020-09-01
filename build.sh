#!/bin/bash
pushd themes/organic
npm i -g postcss-cli autoprefixer tailwindcss
npm i
popd

hugo --gc --minify

ls -lR public/css

