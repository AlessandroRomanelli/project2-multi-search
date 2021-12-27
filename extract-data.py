import os
import sys
from ast import parse, ClassDef, FunctionDef, get_docstring, Constant, AsyncFunctionDef, Module

import pandas as pd


def check_if_py(filePath):
    return os.path.splitext(filePath)[-1] == ".py"


def main():
    path = process_input()
    results = []
    for p, dirs, files in os.walk(path):
        py_files = [f"{p}/{x}" for x in files if check_if_py(f"{p}/{x}")]
        if not len(py_files):
            continue
        for file in py_files:
            results.append(process_file(file))
    pd.concat(results).reset_index(drop=True).to_csv("./results.csv", index=False)


def is_blacklist(node):
    name = node.name.lower()
    return name[0] == "_" or any([x in name for x in ["main", "test"]])


def get_comment(node):
    try:
        comment = get_docstring(node, clean=True)
        if comment:
            return comment.replace('\n', ' ')
    except TypeError as e:
        print(e)
    comments = []
    for x in [y for y in node.body if isinstance(y, Constant) and y.kind is None and len(str(y.value)) > 50]:
        comments.append(str(x.value))
    return " ".join(comments).replace('\n', ' ')


def get_classes(node: Module, path: str):
    data = pd.DataFrame()
    for classNode in [x for x in node.body if isinstance(x, ClassDef) and not is_blacklist(x)]:
        df = pd.DataFrame([{"name": classNode.name, "file": path, "line": classNode.lineno, "type": "class",
                            "comment": get_comment(node)}])
        data = pd.concat([data, df])

        for fn in [x for x in classNode.body if isinstance(x, (FunctionDef, AsyncFunctionDef)) and not is_blacklist(x)]:
            df = pd.DataFrame(
                [{"name": fn.name, "file": path, "line": fn.lineno, "type": "method",
                  "comment": get_comment(fn)}])
            data = pd.concat([data, df])
    return data


def get_functions(node: Module, path: str):
    data = pd.DataFrame()
    for functionNode in [x for x in node.body if
                         isinstance(x, (FunctionDef, AsyncFunctionDef)) and not is_blacklist(x)]:
        df = pd.DataFrame(
            [{"name": functionNode.name, "file": path, "line": functionNode.lineno, "type": "function",
              "comment": get_comment(functionNode)}])
        data = pd.concat([data, df])
    return data


def process_input():
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide at least one parameter <dir>")
    path = sys.argv[1]
    if not os.path.exists(path):
        raise RuntimeError("Invalid path provided")
    return path


def process_file(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        tree = parse(f.read())
    return pd.concat([get_classes(tree, path), get_functions(tree, path)]).reset_index(drop=True).drop_duplicates(
        subset=["name", "comment"])


if __name__ == "__main__":
    main()
    res = pd.read_csv("./results.csv")
    print(len(res["file"].unique()))
    print(len(res[res["type"] == "method"]))
    print(len(res[res["type"] == "function"]))
    print(len(res[res["type"] == "class"]))
