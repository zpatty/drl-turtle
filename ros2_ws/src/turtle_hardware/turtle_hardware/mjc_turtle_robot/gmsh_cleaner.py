"""
Based on the GMSH Converter by @mohammad200h
https://github.com/mohammad200h/GMSHConverter
"""

import argparse
from collections import Counter
import dataclasses
import io
import os
import pathlib
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import gmsh


@dataclasses.dataclass
class Elements:
    """Stores elements extracted from GMSH model."""

    def __init__(
        self,
        element_types: Dict[Tuple[int, int], np.ndarray],
        element_tags: Dict[Tuple[int, int], np.ndarray],
        element_node_tags: Dict[Tuple[int, int], np.ndarray],
    ):
        self.types = element_types
        self.tags = element_tags
        self.node_tags = element_node_tags


@dataclasses.dataclass
class Nodes:
    """Stores nodes extracted from GMSH model."""

    def __init__(self, indexes: np.ndarray, nodes: np.ndarray):
        self.indexes = indexes
        self.nodes = nodes


@dataclasses.dataclass
class Obj:
    """Stores faces, nodes and entities extracted from GMSH model."""

    def __init__(
        self,
        name: str,
        entities: List[Tuple[int, int]],
        elements: Elements,
        nodes: Nodes,
    ):
        self.name = name
        self.elements = elements
        self.nodes = nodes
        self.entities = entities


def _has_msh_extension(filename: str):
    filename_without_extension, extension = os.path.splitext(filename)
    return filename_without_extension, extension.lower() == ".msh"


def _split_path_and_filename(path: pathlib.Path):
    directory, filename = os.path.split(path)
    return directory + os.sep, filename


def _get_path_info(path: pathlib.Path):
    path, filename = _split_path_and_filename(path)
    filename_without_extension, _ = _has_msh_extension(filename)
    return path, filename_without_extension


def _get_nodes():
    nodes_index, points, _ = gmsh.model.mesh.getNodes()
    return nodes_index, points.reshape(-1, 3)


def _is_there_a_shared_surface(
    entity_one: Tuple[int, int], entity_two: Tuple[int, int]
) -> bool:
    """Check if there is a shared surface between two volume entities."""
    entities = [entity_one, entity_two]
    e_surface = _get_boundary_entities_for_volumes(entities)
    surface = []
    for e in e_surface:
        # Shared surface will be repeated but one will have opposite sign.
        surface.append(abs(e[1]))
    occ = _get_maximum_occurrence(surface)
    return occ > 1


def _create_adjacency_matrix(volume_entities: List[Tuple[int, int]]):
    num_vol = len(volume_entities)
    adjacency_matrix = np.zeros((num_vol, num_vol))
    for i, v_e_one in enumerate(volume_entities):
        for j, v_e_two in enumerate(volume_entities):
            if i != j:
                adjacency_matrix[i, j] = (
                    1 if _is_there_a_shared_surface(v_e_one, v_e_two) else 0
                )
    return adjacency_matrix


def _create_graph_from_adjacency_matrix(adjacency_matrix: np.ndarray):
    return nx.from_numpy_array(adjacency_matrix)


def _get_objects_from_graph(graph, volume_entities: List[Tuple[int, int]]):
    graphs = list(nx.connected_components(graph))
    objects = []
    for g in graphs:
        v_entities = [volume_entities[i] for i in g]
        objects.append(v_entities)
    return objects


def _get_maximum_occurrence(surfaces_index: List[int]):
    d = Counter(surfaces_index)
    max_occurrence = 0
    for s in surfaces_index:
        occ = d[s]
        if occ > max_occurrence:
            max_occurrence = occ
    return max_occurrence


def _get_all_entities():
    return gmsh.model.getEntities()


def _get_volume_entities(entities: List[Tuple[int, int]]):
    e_volume = []
    for e in entities:
        if e[0] == 3:
            e_volume.append(e)
    return e_volume


def _get_boundary_entities_for_volumes(volume_entities: List[Tuple[int, int]]):
    e_surface = []
    for e_v in volume_entities:
        e_surface += gmsh.model.getBoundary([e_v])
    return e_surface


def _get_normal(n_0: np.ndarray, n_1: np.ndarray, n_2: np.ndarray):
    edge_one = n_1 - n_0
    edge_two = n_2 - n_0
    normal = np.cross(edge_one, edge_two)
    normal /= np.linalg.norm(normal)
    return normal


def _get_normals(elements: np.ndarray, nodes: np.ndarray):
    """Compute normals for the object."""
    normals = []
    for elem in elements:
        n_0 = nodes[elem[0] - 1]
        n_1 = nodes[elem[1] - 1]
        n_2 = nodes[elem[2] - 1]
        normal = _get_normal(n_0, n_1, n_2)
        normals.append(normal)
    return np.array(normals).reshape(-1, 3)


class VolumeExtractor:
    """Class for creating GMSH model from volume entities."""

    def __init__(self, input_file_path: str, output_file_path: str, version: str):
        self.version = version
        self.path, self.file_name = _get_path_info(pathlib.Path(output_file_path))
        input_path = pathlib.Path(input_file_path)
        gmsh.open(input_path.as_posix())

    def process(self):
        entities = _get_all_entities()
        v_entities = _get_volume_entities(entities)

        if len(v_entities) == 0:
            raise ValueError("No volume entities found.")

        # creating graph of connected volumes
        adjacency_matrix = _create_adjacency_matrix(v_entities)
        graph = _create_graph_from_adjacency_matrix(adjacency_matrix)

        objects_per_graph = _get_objects_from_graph(graph, v_entities)
        there_are_separate_objects = len(objects_per_graph) > 1
        objects = []
        for i, v_entities in enumerate(objects_per_graph):
            prefix = None
            if there_are_separate_objects:
                prefix = i + 1

            obj = self.process_object(v_entities, prefix)
            objects.append(obj)

        for obj in objects:
            self.create_model(obj)

    def process_object(self, volume_entities: List[Tuple[int, int]], prefix: int):
        """Create an Object from volume entities."""
        element_types = {}
        element_tags = {}
        element_node_tags = {}

        for e in volume_entities:
            (
                element_types[e],
                element_tags[e],
                element_node_tags[e],
            ) = gmsh.model.mesh.getElements(e[0], e[1])

        v_nodes_data = _get_nodes()
        node_indexes, nodes = v_nodes_data

        obj = Obj(
            self.file_name + "_vol" + (str(prefix) if prefix else ""),
            volume_entities,
            Elements(element_types, element_tags, element_node_tags),
            Nodes(node_indexes, nodes.flatten().tolist()),
        )
        return obj

    def create_model(self, obj: Obj):
        """Write GMSH file and obj file."""
        gmsh.model.add(obj.name)
        gmsh.model.addDiscreteEntity(3, 1)
        gmsh.model.mesh.addNodes(3, 1, obj.nodes.indexes, obj.nodes.nodes)

        for e in obj.entities:
            gmsh.model.mesh.addElements(
                3,
                1,
                obj.elements.types[e],
                obj.elements.tags[e],
                obj.elements.node_tags[e],
            )

        gmsh.model.mesh.reclassifyNodes()
        if self.version == "2.2":
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        elif self.version == "4.1":
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(pathlib.Path(self.path) / (obj.name + ".msh")))


class SurfaceExtractor:
    """Class for creating GMSH model from surface entities."""

    def __init__(self, input_file_path: str, output_file_path: str, version: str):
        self.version = version
        self.path, self.file_name = _get_path_info(pathlib.Path(output_file_path))
        input_path = pathlib.Path(input_file_path)
        gmsh.open(input_path.as_posix())

    def process(self):
        entities = _get_all_entities()
        v_entities = _get_volume_entities(entities)

        if len(v_entities) == 0:
            raise ValueError("No volume entities found.")

        # creating graph of connected volumes
        adjacency_matrix = _create_adjacency_matrix(v_entities)
        graph = _create_graph_from_adjacency_matrix(adjacency_matrix)

        objects_per_graph = _get_objects_from_graph(graph, v_entities)
        there_are_separate_objects = len(objects_per_graph) > 1

        objects = []
        for i, v_entities in enumerate(objects_per_graph):
            prefix = None
            if there_are_separate_objects:
                prefix = i + 1

            obj = self.process_object(v_entities, prefix)
            objects.append(obj)

        for obj in objects:
            self.create_model(obj)

    def process_object(self, volume_entities: List[Tuple[int, int]], prefix: int):
        """Create an Object from surface entities."""
        # getting surfaces around the volume entities
        e_surfaces = gmsh.model.getBoundary(volume_entities)

        element_types = {}
        element_tags = {}
        element_node_tags = {}

        for e_surface in e_surfaces:
            (
                element_types[e_surface],
                element_tags[e_surface],
                element_node_tags[e_surface],
            ) = gmsh.model.mesh.getElements(e_surface[0], abs(e_surface[1]))

        v_nodes_data = _get_nodes()
        node_indexes, nodes = v_nodes_data

        obj = Obj(
            self.file_name + "_surf" + (str(prefix) if prefix else ""),
            e_surfaces,
            Elements(element_types, element_tags, element_node_tags),
            Nodes(node_indexes, nodes.flatten().tolist()),
        )
        return obj

    def create_model(self, obj: Obj):
        """Write GMSH file and obj file."""
        # generating GMSH
        gmsh.model.add(obj.name)
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, obj.nodes.indexes, obj.nodes.nodes)

        for e in obj.entities:
            elements_node_tags = obj.elements.node_tags[e]
            if e[1] < 0:
                elements_node_tags = [np.flip(obj.elements.node_tags[e][0])]

            gmsh.model.mesh.addElements(
                2, 1, obj.elements.types[e], obj.elements.tags[e], elements_node_tags
            )

        gmsh.model.mesh.reclassifyNodes()
        if self.version == "2.2":
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        elif self.version == "4.1":
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(pathlib.Path(self.path) / (obj.name + ".msh")))

        # generate OBJ
        _, nodes = _get_nodes()
        _, _, elem_nodes_index = gmsh.model.mesh.getElements()
        elements = np.array(elem_nodes_index, dtype=np.int32).reshape(-1, 3)

        normals = _get_normals(elements, nodes)

        out = io.StringIO()

        # nodes
        for node in nodes:
            x, y, z = node
            out.write(f"v {x} {y} {z}\n")

        # normals
        for normal in normals:
            x, y, z = normal
            out.write(f"vn {x} {y} {z}\n")

        # faces
        for face in elements:
            i, j, k = face
            out.write(f"f {i}/{i}/{i} {j}/{j}/{j} {k}/{k}/{k}\n")

        with open(pathlib.Path(self.path) / pathlib.Path(obj.name + ".obj"), "w") as f:
            f.write(out.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-b",
        "--binary",
        type=int,
        default=0,
        help="Format Binary or ASCII. For binary set value to 1.",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="4.1",
        help="GMSH file format 4.1 or 2.2. set the value to either 4.1 or 2.2",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the msh file produced by Gmsh App.",
    )
    parser.add_argument("-o", "--output", type=str, help="Path to the output file.")

    args = parser.parse_args()

    # initialize Gmsh
    if args.binary == 1:
        gmsh.initialize()
    else:
        gmsh.initialize(argv=["", "-bin"])

    # produce files only containing volume
    VolumeExtractor(args.input, args.output, args.version).process()

    # produce a files only containing surface
    SurfaceExtractor(args.input, args.output, args.version).process()

    # finalize Gmsh
    gmsh.finalize()


if __name__ == "__main__":
    main()
