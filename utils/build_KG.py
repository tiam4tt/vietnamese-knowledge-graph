"""
Knowledge Graph Builder for Neo4j
==================================
Script to build and populate a Neo4j knowledge graph from extracted entities and relationships.
Includes utilities for visualization, schema inspection, and data cleaning.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from pyvis.network import Network
from neo4j import GraphDatabase
from dotenv import load_dotenv


class Neo4jLoader:
    """Handles loading data into Neo4j database."""
    
    def __init__(self, uri: str, auth: Optional[Tuple[str, str]] = None):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            auth: Optional tuple of (username, password). None for no authentication.
        """
        try:
            if auth is None:
                self.driver = GraphDatabase.driver(uri)
            else:
                self.driver = GraphDatabase.driver(uri, auth=auth)
            self.verify_connection()
        except Exception as e:
            print(f"Initialization Error: {e}")
            self.driver = None

    def verify_connection(self):
        """Verify Neo4j connection is working."""
        try:
            self.driver.verify_connectivity()
            print("Connected to Neo4j successfully.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("1. Is the Neo4j database running?")
            print("2. Are the URI and PORT correct? (default: bolt://localhost:7687)")
            print("3. Is the password correct?")
            raise e

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def load_data(self, documents: List[Dict[str, Any]]):
        """
        Load graph documents into Neo4j.
        
        Args:
            documents: List of graph documents with 'nodes' and 'relationships' keys
        """
        if not self.driver:
            return
            
        with self.driver.session() as session:
            for i, doc in enumerate(documents):
                # Create nodes
                for node in doc['nodes']:
                    # Remove "(Phụ Lục ...)" appendix references from nodes
                    node['id'] = node['id'].split(' (Phụ Lục')[0].strip()
                    
                    # Sanitize label to ensure it's a valid Neo4j label
                    label = "".join(x for x in node['type'] if x.isalnum() or x == "_").upper()
                    if not label:
                        label = "ENTITY"
                    
                    query = (
                        f"MERGE (n:`{label}` {{id: $id}}) "
                        "SET n.type = $type "
                        "SET n += $properties"
                    )
                    session.run(
                        query,
                        id=node['id'],
                        type=node['type'],
                        properties=node.get('properties', {})
                    )
                
                # Create relationships
                for rel in doc['relationships']:
                    rel_type = "".join(x for x in rel['type'] if x.isalnum() or x == "_").upper()
                    if not rel_type:
                        rel_type = "RELATED_TO"
                    
                    # Clean source/target to match nodes
                    source_id = rel['source'].split(' (Phụ Lục')[0].strip()
                    target_id = rel['target'].split(' (Phụ Lục')[0].strip()

                    query = (
                        "MATCH (a {id: $source}), (b {id: $target}) "
                        f"MERGE (a)-[r:`{rel_type}`]->(b) "
                        "SET r += $properties"
                    )
                    session.run(
                        query,
                        source=source_id,
                        target=target_id,
                        properties=rel.get('properties', {})
                    )
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(documents)} documents...", end="\r")
            
            print(f"\nLoaded {len(documents)} documents into Neo4j.")


def print_neo4j_schema(uri: str, auth: Optional[Tuple[str, str]]):
    """
    Print the full schema of the Neo4j database.
    
    Args:
        uri: Neo4j connection URI
        auth: Optional authentication tuple
    """
    driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
    try:
        with driver.session() as session:
            result = session.run("CALL db.schema.visualization()")
            nodes = set()
            relationships = set()
            for record in result:
                for node in record['nodes']:
                    nodes.add(tuple(node.labels))
                for rel in record['relationships']:
                    relationships.add((rel.start_node.labels, rel.type, rel.end_node.labels))
            
            print("Node Labels:")
            for label in sorted(nodes):
                print(f" - {label}")
            print("\nRelationships:")
            for start_labels, rel_type, end_labels in sorted(relationships):
                print(f" - ({start_labels})-[:{rel_type}]->({end_labels})")
    except Exception as e:
        print(f"Error fetching schema: {e}")
    finally:
        driver.close()


def get_neo4j_statistics(uri: str, auth: Optional[Tuple[str, str]]) -> Dict[str, int]:
    """
    Get statistics about the Neo4j database.
    
    Args:
        uri: Neo4j connection URI
        auth: Optional authentication tuple
        
    Returns:
        Dictionary with node_count and relationship_count
    """
    driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
    stats = {}
    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            stats['node_count'] = node_count
            stats['relationship_count'] = rel_count
    except Exception as e:
        print(f"Error fetching statistics: {e}")
    finally:
        driver.close()
    return stats


def find_orphans_neo4j(uri: str, auth: Optional[Tuple[str, str]], n: int = 10):
    """
    Find orphan nodes (nodes with no relationships) in Neo4j.
    
    Args:
        uri: Neo4j connection URI
        auth: Optional authentication tuple
        n: Number of orphan nodes to display
    """
    driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
    query = f"""
    MATCH (n)
    WHERE NOT (n)--()
    RETURN n.id AS id, n.type AS type
    LIMIT {n}
    """
    
    try:
        with driver.session() as session:
            result = session.run(query)
            orphans = [record['id'] for record in result]
            
            # Get count
            count_result = session.run("MATCH (n) WHERE NOT (n)--() RETURN count(n) as count")
            total = count_result.single()['count']
            
            print(f"Total orphan nodes in DB: {total}")
            if orphans:
                print(f"First {n} orphan nodes:")
                for oid in orphans:
                    print(f" - {oid}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()


def remove_orphans_neo4j(uri: str, auth: Optional[Tuple[str, str]]):
    """
    Remove orphan nodes (nodes with no relationships) from Neo4j.
    
    Args:
        uri: Neo4j connection URI
        auth: Optional authentication tuple
    """
    driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
    query = "MATCH (n) WHERE NOT (n)--() DELETE n"
    
    try:
        with driver.session() as session:
            # Check count before
            count_before = session.run(
                "MATCH (n) WHERE NOT (n)--() RETURN count(n) as count"
            ).single()['count']
            print(f"Orphans before deletion: {count_before}")
            
            if count_before > 0:
                session.run(query)
                print(f"Successfully deleted {count_before} orphan nodes.")
            else:
                print("No orphan nodes found to delete.")
                
    except Exception as e:
        print(f"Error removing orphans: {e}")
    finally:
        driver.close()


def query(uri: str, auth: Optional[Tuple[str, str]], cypher_query: str) -> List[Dict[str, Any]]:
    """
    Execute a custom Cypher query on Neo4j.
    
    Args:
        uri: Neo4j connection URI
        auth: Optional authentication tuple
        cypher_query: Cypher query string to execute
        
    Returns:
        List of result records as dictionaries
    """
    driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
    results = []
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            for record in result:
                results.append(record.data())
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        driver.close()
    return results


def main():
    """Main execution function for building the knowledge graph."""
    # Load environment variables
    load_dotenv()
    
    # Configuration
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    AUTH = None  # Set to ("neo4j", "password") if authentication is required
    
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "graph_documents.json")
    
    # Options
    run_import = True
    remove_orphan_nodes = True
    
    # Load data
    print("Loading graph documents...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        graph_documents = json.load(f)
    
    print(f"Loaded {len(graph_documents)} documents.")
    
    # Import data into Neo4j
    print("\n====== Import Data ======\n")
    if run_import:
        try:
            loader = Neo4jLoader(URI, AUTH)
            if loader.driver:
                loader.clear_database()
                loader.load_data(graph_documents)
                loader.close()
                print("Import finished.")
        except Exception as e:
            print(f"Import failed: {e}")
            sys.exit(1)
    else:
        print("Skipping import. Set run_import=True to run.")
    
    # Get statistics and schema
    print("\n====== Statistics ======\n")
    stats = get_neo4j_statistics(URI, AUTH)
    print(f"Total Nodes: {stats.get('node_count', 'N/A')}")
    print(f"Total Relationships: {stats.get('relationship_count', 'N/A')}")
    
    print("\n====== Full Schema ======\n")
    print_neo4j_schema(URI, AUTH)
    
    # Find and remove orphan nodes
    if remove_orphan_nodes:
        print("\n====== Remove Orphan Nodes ======\n")
        find_orphans_neo4j(URI, AUTH, 5)
        remove_orphans_neo4j(URI, AUTH)
    
    print("\n====== Knowledge Graph Build Complete ======")


if __name__ == "__main__":
    main()
