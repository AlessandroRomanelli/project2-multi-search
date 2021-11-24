import os
import sys
from ast import parse, NodeVisitor, ClassDef, Assign, FunctionDef, get_docstring
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
    return node.name[0] == "_" or node.name == "main" or "test" in node.name.lower()


def get_comment(node):
    comment = get_docstring(node, clean=True)
    if not comment:
        return None
    return get_docstring(node, clean=True).replace("\n", "\\n")


class Visitor(NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.data = pd.DataFrame(columns=["name", "file", "line", "type", "comment"])

    def visit_ClassDef(self, node: ClassDef):
        if is_blacklist(node):
            return
        df = pd.DataFrame([{"name": node.name, "file": self.path, "line": node.lineno, "type": "class",
                            "comment": get_comment(node)}])
        self.data = pd.concat([self.data, df])

        for fn in [x for x in node.body if type(x) is FunctionDef]:
            if is_blacklist(fn):
                continue
            df = pd.DataFrame(
                [{"name": fn.name, "file": self.path, "line": fn.lineno, "type": "function",
                  "comment": get_comment(fn)}])
            self.data = pd.concat([self.data, df])

    def visit_FunctionDef(self, node: FunctionDef):
        if is_blacklist(node):
            return
        df = pd.DataFrame(
            [{"name": node.name, "file": self.path, "line": node.lineno, "type": "function",
              "comment": get_comment(node)}])
        self.data = pd.concat([self.data, df])
        pass


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
    visitor = Visitor(path)
    visitor.visit(tree)
    return visitor.data.reset_index(drop=True)


if __name__ == "__main__":
    main()
