#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the entry point for the command-line interface (CLI) application.

It can be used as a handy facility for running the task from a command line.

.. currentmodule:: anypp.cli
.. moduleauthor:: Morten Enemark Lund <mel@anybodytech.com>
"""
import sys
from xml.etree import ElementTree
from typing import Iterable, Optional, TextIO, Any, Dict, Tuple
from ast import literal_eval
import re
from getpass import getpass

import numpy as np

import logging
import click
from .__init__ import __version__

LOGGING_LEVELS = {
    0: logging.NOTSET,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}  #: a mapping of `verbose` option counts to logging levels


class Info(object):
    """An information object to pass data between CLI functions."""

    def __init__(self):  # Note: This object must have an empty constructor.
        """Create a new instance."""
        self.verbose: int = 0


# pass_info is a decorator for functions that pass 'Info' objects.
#: pylint: disable=invalid-name
pass_info = click.make_pass_decorator(Info, ensure=True)


# Change the options to below to suit the actual options for your task (or
# tasks).
@click.group()
@click.option("--verbose", "-v", count=True, help="Enable verbose output.")
@pass_info
def cli(info: Info, verbose: int):
    """Run anypp."""
    # Use the verbosity count to determine the logging level...
    if verbose > 0:
        logging.basicConfig(
            level=LOGGING_LEVELS[verbose]
            if verbose in LOGGING_LEVELS
            else logging.DEBUG
        )
        click.echo(
            click.style(
                f"Verbose logging is enabled. "
                f"(LEVEL={logging.getLogger().getEffectiveLevel()})",
                fg="yellow",
            )
        )
    info.verbose = verbose


@cli.command()
@click.argument('fp', type=click.File(), required = False)
@pass_info
def convert(_: Info, fp):
    """Convert a file between from meshlab pp to anyscript"""

    if fp:
        data = fp.read()
    else:
        data = click.prompt("Paste pp file or anyscript lines", hide_input=True)
        lines = [data]
        while True:
            line = getpass("\033[F")
            if line:
                lines.append(line)
            else:
                break
        data = '\n'.join(lines)

    print("\n\n")

    if data.startswith("<!DOCTYPE PickedPoints>"):

        points = parse_pp_file(data)
        print("\n\n Point cloud:")
        print("================================")
        click.echo(format_anyscript_pointcloud(points))

        print("\n\n Anyscript values:")
        print("================================")
        click.echo(format_anyscript(points))

    elif "=" in data and  data.strip().startswith("{"):
        points = parse_anyanyscript(data)
        click.echo(format_ppfile(points))



@cli.command()
def version():
    """Get the library version."""
    click.echo(click.style(f"{__version__}", bold=True))



def parse_pp_file(data: str):

    try:
        etree = ElementTree.fromstring(data)
    except ElementTree.ParseError as e:
        raise ValueError(f"Could not parse XML file: {e}")
    allpoints = {}
    for elem in etree.iter("point"):
        try:
            point = np.array(
                (float(elem.attrib["x"]), float(elem.attrib["y"]), float(elem.attrib["z"]))
            )
        except ValueError:
            continue
        allpoints[elem.attrib["name"]] = point
    return allpoints


def signal_last(it:Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield False, ret_var
        ret_var = val
    yield True, ret_var




def format_anyscript_pointcloud(points: Dict[str, Any]):
    anyscript = ""
    first = True
    for is_last, (name, val) in signal_last(points.items()):
        # replace invalid characters in name with underscores
        name = "".join(c if c.isalnum() else "_" for c in name)
        comma = "" if is_last else ","
        anyscript += f"{{{val[0]}, {val[1]}, {val[2]}}}{comma} // {name} \n"
    return anyscript


def format_anyscript(points: Dict[str, Any]):
    anyscript = ""
    for name, val in points.items():
        # replace invalid characters in name with underscores
        name = "".join(c if c.isalnum() else "_" for c in name)
        anyscript += f"AnyFloat {name} = {{{val[0]},{val[1]}, {val[2]}}};\n"
    return anyscript



ANYSCRIPT_POINT_LINE = re.compile(r'\s+({.+})\s*,?\s*[//]?\W*(.*)')



def parse_anyanyscript(data: str):
    points: Dict[str, Any] = {}
    for line in data.splitlines():
        m = ANYSCRIPT_POINT_LINE.match(line)
        if m:
            val, name = m.groups()
            try:
                points[name] = _parse_anyscript(val)
            except (SyntaxError, ValueError):
                continue
    return points
    

def format_ppfile(points: Dict[str, Any]):
    output = ""
    for name, val in points.items():
        output += f'<point x="{val[0]}" y="{val[1]}" z="{val[2]}" active="1" name="{name}" />\n'
    return output


def _recursive_replace(iterable: Iterable, old:Any, new:Any):
    for i, elem in enumerate(iterable):
        if isinstance(elem, list):
            _recursive_replace(elem, old, new)
        elif elem == old:
            iterable[i] = new


TRIPEL_QUOTE_WRAP = re.compile(r'([^\[\]",\s]+)')


def _parse_anyscript(val: str):
    """Convert a str AnyBody data repr into Numpy array."""
    if val.startswith("{") and val.endswith("}"):
        val = val.replace("{", "[").replace("}", "]")
    if val == "[...]":
        val = '"..."'
    try:
        out = literal_eval(val)
    except (SyntaxError, ValueError):
        try:
            if "nan," in val:
                # handle the case where AnyBody has output 'nan' values
                val2 = val.replace("nan,", ' "nan",')
                out = literal_eval(val2)
                _recursive_replace(out, "nan", float("nan"))
            else:
                raise SyntaxError
        except (SyntaxError, ValueError):
            val, _ = TRIPEL_QUOTE_WRAP.subn(r"'''\1'''", val)
            if val == "":
                val = "None"
            if val.startswith('"') and val.endswith('"'):
                val = "'''" + val[1:-1] + r"'''"
            out = literal_eval(val)
    if isinstance(out, list):
        out = np.array(out)
    return out