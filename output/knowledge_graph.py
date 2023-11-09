import networkx as nx
import matplotlib.pyplot as plt

class knowledgeGraph():
    def __init__(self):
        """
        Initialize the knowledge graph
        """
        self.G = nx.MultiDiGraph()
        
    def draw_graph(self, all_relations : list, all_data : list, store_graph : bool, save_path : str = None):
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


        # Visualize the graph
        pos = nx.spring_layout(self.G, seed= 42)
        labels = {n: str(n) for n in self.G.nodes()}
        edge_labels = {(a, b): labels['label'] for a, b, labels in self.G.edges(data=True)}

        plt.figure(figsize=(12, 10))
        nx.draw(self.G, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue", edge_color='green')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)
            
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