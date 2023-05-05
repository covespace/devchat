from dataclasses import asdict
import os
from typing import List
from xml.etree.ElementTree import ParseError
import networkx as nx
from devchat.prompt import Prompt
from devchat.utils import store_to_git


class Store:
    def __init__(self, path: str):
        """
        Initializes a Store instance.

        Args:
            path (str): The path to the file containing the store.
        """
        self._path = os.path.expanduser(path)
        if os.path.isfile(self._path):
            try:
                self._graph = nx.read_graphml(self._path)
            except ParseError as error:
                raise ValueError(f"Invalid file format for store: {self._path}") from error
        else:
            self._graph = nx.DiGraph()

    @property
    def path(self) -> str:
        """
        The path to the file containing the store.
        """
        return self._path

    def store_prompt(self, prompt: Prompt) -> str:
        """
        Store a prompt in the store.

        Args:
            prompt (Prompt): The prompt to store.
        """
        prompt_hash = store_to_git(asdict(prompt))
        # Add the prompt to the graph
        self._graph.add_node(prompt.hash, timestamp=prompt.timestamp)

        # Add edges for parents and references
        for parent_hash in prompt.parents:
            if parent_hash not in self._graph:
                raise ValueError(f'Parent {parent_hash} not found in the store.')
            self._graph.add_edge(prompt.hash, parent_hash)
        for reference_hash in prompt.references:
            if reference_hash not in self._graph:
                raise ValueError(f'Reference {reference_hash} not found in the store.')
            self._graph.add_edge(prompt_hash, reference_hash)
        nx.write_graphml(self._graph, self._path)
        return prompt_hash

    def get_prompt(self, prompt_hash: str) -> dict:
        """
        Retrieve a prompt from the store.

        Args:
            prompt_hash (str): The hash of the prompt to retrieve.
        Returns:
            Prompt: The retrieved prompt.
        """
        if prompt_hash not in self._graph:
            raise ValueError(f'Prompt {prompt_hash} not found in the store.')
        return self._graph.nodes[prompt_hash]

    def select_recent(self, start: int, end: int) -> List[str]:
        """
        Select recent prompts.

        Args:
            start (int): The start index.
            end (int): The end index (excluded).
        Returns:
            List[Prompt]: The list of prompts selected.
                          If end is greater than the number of all prompts,
                          the list will contain prompts from start to the end of the list.
        """
        sorted_nodes = sorted(self._graph.nodes(data=True),
                              key=lambda x: x[1]['timestamp'],
                              reverse=True)
        if end > len(sorted_nodes):
            end = len(sorted_nodes)
        return [note[0] for note in sorted_nodes[start:end]]
