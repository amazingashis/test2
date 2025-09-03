import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from collections import defaultdict, Counter
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationshipGraph:
    """Manages relationship graphs for semantic RAG"""
    
    def __init__(self):
        """Initialize the relationship graph"""
        self.graph = nx.DiGraph()
        self.node_types = {}
        self.edge_types = {}
        self.metadata = {}
    
    def add_node(self, node_id: str, node_type: str, **attributes) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id (str): Unique identifier for the node
            node_type (str): Type of the node (e.g., 'document', 'field', 'concept')
            **attributes: Additional node attributes
        """
        self.graph.add_node(node_id, node_type=node_type, **attributes)
        self.node_types[node_id] = node_type
        
        logger.debug(f"Added node: {node_id} (type: {node_type})")
    
    def add_relationship(self, source: str, target: str, relationship_type: str, 
                        weight: float = 1.0, **attributes) -> None:
        """
        Add a relationship (edge) between two nodes.
        
        Args:
            source (str): Source node ID
            target (str): Target node ID
            relationship_type (str): Type of relationship
            weight (float): Relationship strength/weight
            **attributes: Additional edge attributes
        """
        self.graph.add_edge(
            source, target, 
            relationship_type=relationship_type,
            weight=weight,
            **attributes
        )
        
        edge_key = (source, target)
        self.edge_types[edge_key] = relationship_type
        
        logger.debug(f"Added relationship: {source} --[{relationship_type}]--> {target} (weight: {weight})")
    
    def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """
        Get neighbors of a node, optionally filtered by relationship type.
        
        Args:
            node_id (str): Node ID to get neighbors for
            relationship_type (str, optional): Filter by relationship type
            
        Returns:
            List[str]: List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []
        
        neighbors = []
        
        # Outgoing edges
        for neighbor in self.graph.successors(node_id):
            edge_data = self.graph[node_id][neighbor]
            if relationship_type is None or edge_data.get('relationship_type') == relationship_type:
                neighbors.append(neighbor)
        
        # Incoming edges
        for neighbor in self.graph.predecessors(node_id):
            edge_data = self.graph[neighbor][node_id]
            if relationship_type is None or edge_data.get('relationship_type') == relationship_type:
                neighbors.append(neighbor)
        
        return list(set(neighbors))  # Remove duplicates
    
    def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """
        Find paths between two nodes.
        
        Args:
            source (str): Source node ID
            target (str): Target node ID
            max_length (int): Maximum path length
            
        Returns:
            List[List[str]]: List of paths (each path is a list of node IDs)
        """
        try:
            # Convert to undirected for path finding
            undirected_graph = self.graph.to_undirected()
            
            paths = []
            for path in nx.all_simple_paths(undirected_graph, source, target, cutoff=max_length):
                paths.append(path)
            
            return paths
        except Exception as e:
            logger.error(f"Error finding paths from {source} to {target}: {str(e)}")
            return []
    
    def get_subgraph(self, nodes: List[str]) -> 'RelationshipGraph':
        """
        Extract a subgraph containing specified nodes.
        
        Args:
            nodes (List[str]): List of node IDs to include
            
        Returns:
            RelationshipGraph: New graph containing only specified nodes and their connections
        """
        subgraph_nx = self.graph.subgraph(nodes)
        
        new_graph = RelationshipGraph()
        new_graph.graph = subgraph_nx.copy()
        
        # Copy metadata
        for node in nodes:
            if node in self.node_types:
                new_graph.node_types[node] = self.node_types[node]
        
        for edge in subgraph_nx.edges():
            if edge in self.edge_types:
                new_graph.edge_types[edge] = self.edge_types[edge]
        
        return new_graph
    
    def cluster_nodes(self, algorithm: str = 'louvain') -> Dict[str, int]:
        """
        Cluster nodes in the graph.
        
        Args:
            algorithm (str): Clustering algorithm ('louvain', 'label_propagation')
            
        Returns:
            Dict[str, int]: Mapping from node ID to cluster ID
        """
        try:
            # Convert to undirected for clustering
            undirected_graph = self.graph.to_undirected()
            
            if algorithm == 'louvain':
                import community as community_louvain
                clusters = community_louvain.best_partition(undirected_graph)
            elif algorithm == 'label_propagation':
                cluster_generator = nx.algorithms.community.label_propagation_communities(undirected_graph)
                clusters = {}
                for i, cluster in enumerate(cluster_generator):
                    for node in cluster:
                        clusters[node] = i
            else:
                logger.error(f"Unknown clustering algorithm: {algorithm}")
                return {}
            
            logger.info(f"Clustered {len(clusters)} nodes into {len(set(clusters.values()))} clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering nodes: {str(e)}")
            return {}
    
    def get_central_nodes(self, centrality_type: str = 'betweenness', top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get most central nodes in the graph.
        
        Args:
            centrality_type (str): Type of centrality ('betweenness', 'degree', 'pagerank')
            top_k (int): Number of top nodes to return
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, centrality_score) tuples
        """
        try:
            if centrality_type == 'betweenness':
                centrality = nx.betweenness_centrality(self.graph)
            elif centrality_type == 'degree':
                centrality = nx.degree_centrality(self.graph)
            elif centrality_type == 'pagerank':
                centrality = nx.pagerank(self.graph)
            else:
                logger.error(f"Unknown centrality type: {centrality_type}")
                return []
            
            # Sort by centrality score
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_nodes[:top_k]
            
        except Exception as e:
            logger.error(f"Error calculating centrality: {str(e)}")
            return []
    
    def visualize_graph(self, output_file: str = None, layout: str = 'spring', 
                       node_size_attribute: str = None, interactive: bool = True) -> None:
        """
        Visualize the graph.
        
        Args:
            output_file (str, optional): File to save the visualization
            layout (str): Layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_attribute (str, optional): Node attribute to use for sizing
            interactive (bool): Whether to create interactive plot
        """
        if len(self.graph.nodes()) == 0:
            logger.warning("Cannot visualize empty graph")
            return
        
        try:
            if interactive:
                self._create_interactive_visualization(output_file, layout, node_size_attribute)
            else:
                self._create_static_visualization(output_file, layout, node_size_attribute)
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
    
    def _create_interactive_visualization(self, output_file: str, layout: str, 
                                        node_size_attribute: str) -> None:
        """Create interactive Plotly visualization"""
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text
            node_info = self.graph.nodes[node]
            text = f"ID: {node}<br>Type: {node_info.get('node_type', 'unknown')}"
            for key, value in node_info.items():
                if key not in ['node_type', 'pos']:
                    text += f"<br>{key}: {str(value)[:50]}"
            node_text.append(text)
            
            # Node color based on type
            node_type = self.node_types.get(node, 'unknown')
            if node_type == 'document':
                node_color.append('red')
            elif node_type == 'field':
                node_color.append('blue')
            elif node_type == 'concept':
                node_color.append('green')
            else:
                node_color.append('gray')
            
            # Node size
            if node_size_attribute and node_size_attribute in node_info:
                size = float(node_info[node_size_attribute]) * 20
            else:
                size = 10
            node_size.append(size)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = self.graph[edge[0]][edge[1]]
            relationship = edge_data.get('relationship_type', 'unknown')
            weight = edge_data.get('weight', 1.0)
            edge_info.append(f"{edge[0]} --[{relationship}({weight:.2f})]--> {edge[1]}")
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='black')
            )
        ))
        
        fig.update_layout(
            title='Semantic RAG Relationship Graph',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Nodes: Red=Documents, Blue=Fields, Green=Concepts",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Interactive visualization saved to {output_file}")
        else:
            fig.show()
    
    def _create_static_visualization(self, output_file: str, layout: str, 
                                   node_size_attribute: str) -> None:
        """Create static matplotlib visualization"""
        plt.figure(figsize=(12, 8))
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, edge_color='gray')
        
        # Group nodes by type
        node_groups = defaultdict(list)
        for node in self.graph.nodes():
            node_type = self.node_types.get(node, 'unknown')
            node_groups[node_type].append(node)
        
        # Draw nodes by type
        colors = {'document': 'red', 'field': 'blue', 'concept': 'green', 'unknown': 'gray'}
        for node_type, nodes in node_groups.items():
            nx.draw_networkx_nodes(
                self.graph, pos, 
                nodelist=nodes,
                node_color=colors.get(node_type, 'gray'),
                node_size=300,
                alpha=0.7
            )
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.title('Semantic RAG Relationship Graph')
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Static visualization saved to {output_file}")
        else:
            plt.show()
    
    def save_graph(self, file_path: str, format: str = 'json') -> None:
        """
        Save graph to file.
        
        Args:
            file_path (str): Output file path
            format (str): Output format ('json', 'gml', 'graphml')
        """
        try:
            if format == 'json':
                data = {
                    'nodes': [],
                    'edges': [],
                    'metadata': self.metadata
                }
                
                for node in self.graph.nodes(data=True):
                    node_data = {'id': node[0]}
                    node_data.update(node[1])
                    data['nodes'].append(node_data)
                
                for edge in self.graph.edges(data=True):
                    edge_data = {'source': edge[0], 'target': edge[1]}
                    edge_data.update(edge[2])
                    data['edges'].append(edge_data)
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            elif format == 'gml':
                nx.write_gml(self.graph, file_path)
            elif format == 'graphml':
                nx.write_graphml(self.graph, file_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return
            
            logger.info(f"Graph saved to {file_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
    
    def load_graph(self, file_path: str, format: str = 'json') -> None:
        """
        Load graph from file.
        
        Args:
            file_path (str): Input file path
            format (str): Input format ('json', 'gml', 'graphml')
        """
        try:
            if format == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.graph.clear()
                self.node_types.clear()
                self.edge_types.clear()
                
                # Load nodes
                for node_data in data['nodes']:
                    node_id = node_data.pop('id')
                    node_type = node_data.get('node_type', 'unknown')
                    self.add_node(node_id, node_type, **node_data)
                
                # Load edges
                for edge_data in data['edges']:
                    source = edge_data.pop('source')
                    target = edge_data.pop('target')
                    relationship_type = edge_data.get('relationship_type', 'unknown')
                    weight = edge_data.get('weight', 1.0)
                    self.add_relationship(source, target, relationship_type, weight, **edge_data)
                
                self.metadata = data.get('metadata', {})
                
            elif format == 'gml':
                self.graph = nx.read_gml(file_path)
            elif format == 'graphml':
                self.graph = nx.read_graphml(file_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return
            
            logger.info(f"Graph loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing graph statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'is_connected': nx.is_weakly_connected(self.graph),
            'density': nx.density(self.graph),
            'node_types': dict(Counter(self.node_types.values())),
            'edge_types': dict(Counter(self.edge_types.values()))
        }
        
        if stats['num_nodes'] > 0:
            stats['average_degree'] = sum(dict(self.graph.degree()).values()) / stats['num_nodes']
        else:
            stats['average_degree'] = 0
        
        return stats
