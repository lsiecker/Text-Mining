import networkx as nx
import matplotlib.pyplot as plt
import json
import math

class knowledgeGraph():
    def __init__(self):
        """
        Initialize the knowledge graph
        """
        self.G = nx.MultiDiGraph()
        
    def draw_graph(self, all_relations : list, all_data : list, store_graph : bool, filter_labels : list, subset : tuple = (0,5), save_path : str = None):
        """
        Draw the knowledge graph
        :param all_relations: list of relations
        :param all_data: list of entities
        :return: None
        """
        new_relations = []

        for relation in all_relations:
            if relation[2] == 'is_similar_to':
                new_relations.append(relation)
                node_a = relation[0]
                node_b = relation[1]
                for relations in all_relations:
                    if relations[0] == node_a and relations[2] != 'is_similar_to':
                        new_relations.append((node_b, relations[1], relations[2]))
                    else:
                        new_relations.append(relations)
            else:
                new_relations.append(relation)

                
            
        entity_labels = {}

        for data in all_data:
            text = data["value"]["text"]
            labels = data["value"]["labels"]
            entity_labels[text] = labels



        # Add nodes and edges to the graph
        for entity1, entity2, label in new_relations:
            self.G.add_node(entity1, labels=entity_labels.get(entity1, []))
            self.G.add_node(entity2, labels=entity_labels.get(entity2, []))
            self.G.add_edge(entity1, entity2, label=label)

        subset_landmarks = [entity for entity, labels in entity_labels.items() if labels == filter_labels][subset[0]:subset[1]]
        subset_relations = [triple for triple in new_relations if any(item in subset_landmarks for item in triple)]
        subset_entities = [item for triple in subset_relations for item in triple[:2]]

        subgraph = nx.subgraph(self.G, subset_entities)

        # Visualize subgraph
        pos = nx.spring_layout(subgraph, seed= 42, k = 6)
        labels = {n: str(n) for n in subgraph.nodes()}
        edge_labels = {(a, b): labels['label'] for a, b, labels in subgraph.edges(data=True)}

        plt.figure(figsize=(12, 10))
        nx.draw(subgraph, pos, with_labels=True, labels=labels, node_size=500, node_color="lightblue", edge_color='green')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)
            
        if store_graph:
            plt.savefig(save_path, format="PNG")
            print("Graph stored in {}".format(save_path))
            
        plt.show()
        
        print("Query examples:")
        # Query the graph
        print("All nodes (entities):")
        for node in self.G.nodes(data=True):
            print(node)
        print("All edges (relations):")
        for edge in self.G.edges(data=True):
            print(edge)
    
    def export_json(self, all_relations: list, all_data: list, save_path: str = None):
        """
        Export the knowledge graph to a JSON file
        :param save_path: Path to the JSON file
        """ 
        data = {
            "nodes": [
                {"id": node, "labels": labels}
                for node, labels in self.G.nodes(data="labels")
            ],
            "edges": [
                {"source": source, "target": target, "label": label}
                for source, target, label in self.G.edges(data="label")
            ],
        }

        with open(save_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Knowledge graph exported to {save_path}")