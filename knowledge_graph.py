import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.MultiDiGraph()


all_data = [{"id": "tBYLAMaY7Q", "type": "labels", "value": {"end": 18, "text": "Chartres Cathedral", "start": 0, "labels": ["landmark_name"]}},
             {"id": "rXo1WB9tDL", "type": "labels", "value": {"end": 71, "text": "Cathedral of Our Lady of Chartres", "start": 38, "labels": ["landmark_name"]}},
                {"id": "8FJnNiKvM2", "type": "labels", "value": {"end": 116, "text": "France", "start": 110, "labels": ["location"]}},
                 {"id": "3VbdkKDmSx", "type": "labels", "value": {"end": 363, "text": "episcopal", "start": 354, "labels": ["people"]}},
                  {"id": "fWOLrqi2qp", "type": "labels", "value": {"end": 492, "text": "High Gothic and Classic Gothic architecture", "start": 449, "labels": ["type"]}}]

all_relations = [('Chartres Cathedral', 'France', 'LocatedIn'),('Cathedral of Our Lady of Chartres', 'France', 'LocatedIn'),('Chartres Cathedral', 'episcopal', 'HasProperty'),('Chartres Cathedral', 'High Gothic and Classic Gothic architecture', 'HasType'),
                 ('Cathedral of Our Lady of Chartres', 'episcopal', 'HasProperty'),('Cathedral of Our Lady of Chartres', 'High Gothic and Classic Gothic architecture', 'HasTest'),
                 ('Cathedral of Our Lady of Chartres','Chartres Cathedral', 'is_similar_to')]

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
    G.add_node(entity1, labels=entity_labels.get(entity1, []))
    G.add_node(entity2, labels=entity_labels.get(entity2, []))
    G.add_edge(entity1, entity2, label=label)


# Visualize the graph
pos = nx.spring_layout(G, seed= 42)
labels = {n: str(n) for n in G.nodes()}
edge_labels = {(a, b): labels['label'] for a, b, labels in G.edges(data=True)}

plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue", edge_color='green')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)
plt.show()


# Query the graph
print("Nodes:")
for node in G.nodes(data=True):
    print(node)
print("Edges:")
for edge in G.edges(data=True):
    print(edge)


