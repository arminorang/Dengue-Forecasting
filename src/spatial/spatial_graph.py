from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


class SpatialGraphBuilder:
    """Build spatial graphs from municipality geometries"""

    def __init__(self, spatial_dir: Optional[Path] = None):
        self.spatial_dir = spatial_dir

    def construct_spatial_graph(
        self,
        gdf: gpd.GeoDataFrame,
        method: str = "adjacency",
        k: int = 5,
        distance_threshold_km: float = 100.0,
        save_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Construct a spatial graph from municipality polygons."""
        logger.info(f"🔗 Building spatial graph using '{method}' method")

        gdf = self._prepare_geometries(gdf)
        coords = self._compute_centroids(gdf)
        n = len(gdf)

        if method == "adjacency":
            edges, weights = self._build_adjacency_graph(gdf, coords)
        elif method == "knn":
            edges, weights = self._build_knn_graph(coords, k)
        elif method == "distance":
            edges, weights = self._build_distance_graph(coords, distance_threshold_km)
        else:
            raise ValueError(f"Unknown graph method: {method}")

        edge_index = np.array(edges, dtype=int).T if edges else np.zeros((2, 0), dtype=int)
        edge_weight = np.array(weights, dtype=float) if weights else np.array([])
        node_info = self._build_node_table(gdf, coords)

        logger.info(
            f"✅ Graph built: {n} nodes, {edge_index.shape[1]} edges, "
            f"avg degree = {edge_index.shape[1] / n:.2f}"
        )

        if save_name and self.spatial_dir is not None:
            self._save_graph(save_name, edge_index, edge_weight, coords, node_info)

        return edge_index, edge_weight, node_info

    def _prepare_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Ensure valid geometries and projected CRS."""
        gdf = gdf.copy()
        if gdf.crs is None:
            logger.warning("⚠️  GeoDataFrame has no CRS, assuming EPSG:4326")
            gdf.set_crs("EPSG:4326", inplace=True)

        if gdf.crs.is_geographic:
            logger.info("📐 Reprojecting to EPSG:5880 (Brazil Polyconic)")
            gdf = gdf.to_crs(5880)

        gdf["geometry"] = gdf.geometry.buffer(0)
        return gdf

    def _compute_centroids(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        centroids = gdf.geometry.centroid
        return np.column_stack([centroids.x.values, centroids.y.values])

    def _build_adjacency_graph(self, gdf: gpd.GeoDataFrame, coords: np.ndarray):
        """Shared-border adjacency graph"""
        logger.info("🧩 Computing adjacency graph")
        edges, weights = [], []
        
        try:
            sindex = gdf.sindex
        except:
            sindex = None

        for i, geom in enumerate(gdf.geometry):
            if sindex:
                candidates = list(sindex.intersection(geom.bounds))
            else:
                candidates = range(len(gdf))
            
            for j in candidates:
                if j <= i:
                    continue
                if geom.touches(gdf.geometry.iloc[j]):
                    dist_km = np.linalg.norm(coords[i] - coords[j]) / 1000.0
                    edges.extend([[i, j], [j, i]])
                    weights.extend([dist_km, dist_km])

        return edges, weights

    def _build_knn_graph(self, coords: np.ndarray, k: int):
        """k-nearest neighbors graph"""
        logger.info(f"📍 Computing {k}-NN graph")
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        edges, weights = [], []
        for i in range(len(coords)):
            for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                edges.append([i, j])
                weights.append(dist / 1000.0)

        return edges, weights

    def _build_distance_graph(self, coords: np.ndarray, threshold_km: float):
        """Distance-threshold graph"""
        logger.info(f"📏 Computing distance graph (< {threshold_km} km)")
        edges, weights = [], []
        threshold_m = threshold_km * 1000.0

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold_m:
                    edges.extend([[i, j], [j, i]])
                    weights.extend([dist / 1000.0, dist / 1000.0])

        return edges, weights

    def _build_node_table(self, gdf: gpd.GeoDataFrame, coords: np.ndarray) -> pd.DataFrame:
        """Build node metadata table - SIMPLIFIED VERSION"""
        n = len(gdf)
        
        # Since we know your GDF has CD_MUN and NM_MUN, just use them directly
        codes = gdf["CD_MUN"].to_numpy()
        names = gdf["NM_MUN"].to_numpy()

        return pd.DataFrame({
            "node_id": np.arange(n),
            "municipality_code": codes,
            "municipality_name": names,
            "x": coords[:, 0],
            "y": coords[:, 1],
        })

    def _save_graph(self, name: str, edge_index: np.ndarray, edge_weight: np.ndarray, 
                    coords: np.ndarray, node_info: pd.DataFrame):
        """Save graph to disk"""
        if self.spatial_dir is None:
            raise ValueError("spatial_dir not set")

        self.spatial_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            self.spatial_dir / f"{name}.npz",
            edge_index=edge_index,
            edge_weight=edge_weight,
            coords=coords,
        )

        node_info.to_csv(self.spatial_dir / f"{name}_nodes.csv", index=False)
        logger.info(f"💾 Saved to {self.spatial_dir}")

    def visualize_spatial_graph(self, gdf: gpd.GeoDataFrame, edge_index: np.ndarray,
                                output_path: Optional[Path] = None):
        """Visualize graph"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            
            fig, ax = plt.subplots(figsize=(12, 10))
            gdf.boundary.plot(ax=ax, linewidth=0.5, color='gray', alpha=0.5)
            
            coords = np.column_stack([gdf.geometry.centroid.x.values, 
                                     gdf.geometry.centroid.y.values])
            
            if edge_index.shape[1] > 0:
                lines = [[coords[edge_index[0, i]], coords[edge_index[1, i]]] 
                        for i in range(edge_index.shape[1])]
                lc = LineCollection(lines, colors='blue', linewidths=0.5, alpha=0.3)
                ax.add_collection(lc)
            
            ax.scatter(coords[:, 0], coords[:, 1], c='red', s=20, zorder=5, alpha=0.6)
            ax.set_title(f"Spatial Graph: {len(gdf)} Municipalities, {edge_index.shape[1]} Edges")
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return fig, ax
            
        except ImportError:
            logger.warning("matplotlib not installed")
            return None, None