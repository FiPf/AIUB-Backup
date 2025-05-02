use pyo3::prelude::*;
// Remove the explicit import if using prelude::*
// use pyo3::wrap_pyfunction;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use itertools::Itertools; // Make sure itertools is in your Cargo.toml dependencies
use std::collections::HashMap;

// The module definition function
#[pymodule]
#[pyo3(name = "my_dbcv_module")] // Optional: specify the python module name explicitly
fn dbcv(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Correct way to add the function
    m.add_function(wrap_pyfunction!(dbcv_score, m)?)?;
    Ok(())
}

/// Density-Based Clustering Validation (faster version)
/// Returns a cluster validity score in the range of [-1, 1]
#[pyfunction]
fn dbcv_score(
    py: Python, // Acquire GIL token if needed for numpy operations inside
    x: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<i32>,
    _dist_function: Option<String>, // Parameter is unused, marked with _
) -> PyResult<f64> {
    // Access arrays safely using view() or to_owned_array()
    let x_view = x.as_array();
    let labels_view = labels.as_array();

    // It's often better to release the GIL while doing heavy computation
    let result = py.allow_threads(|| {
        let core_dists = precompute_core_dists(&x_view, &labels_view);
        let graph = mutual_reach_graph(&x_view, &core_dists);
        let mst = mutual_reach_mst_edges(&graph);
        clustering_validity(&mst, &labels_view)
    });

    Ok(result)
}

/// Euclidean distance between two points
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute full pairwise distance matrix
fn distance_matrix(x: &ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut dist = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&x.row(i), &x.row(j));
            dist[(i, j)] = d;
            dist[(j, i)] = d;
        }
    }
    dist
}

/// Precompute core distances per point per cluster
fn precompute_core_dists(x: &ArrayView2<f64>, labels: &ArrayView1<i32>) -> Vec<f64> {
    let n = x.nrows();
    let dims = x.ncols();
    let mut cores = vec![0.0; n]; // Initialize with 0.0
    let mut members: HashMap<i32, Vec<usize>> = HashMap::new(); // Explicit type

    for (i, &lab) in labels.iter().enumerate() {
        // Skip noise points if represented by a specific label like -1
        if lab < 0 { continue; }
        members.entry(lab).or_insert_with(Vec::new).push(i);
    }

    for (_lab, indices) in members.iter() { // Use iter() instead of values() to avoid borrow issues if needed elsewhere
        if indices.len() <= 1 { // Handle clusters with 0 or 1 point
             for &idx in indices {
                 cores[idx] = f64::INFINITY; // Or 0.0, depending on desired behavior
             }
             continue;
        }

        // Select returns an owned Array, use view() for ArrayView2
        let sub = x.select(Axis(0), indices);
        let dm = distance_matrix(&sub.view()); // Pass view

        for &i in indices {
            // Find the corresponding row index within the sub-matrix 'dm'
            let local_idx = indices.iter().position(|&v| v == i).unwrap(); // Safe because i is in indices

            // Get distances from the dense sub-matrix
            let dists: Vec<f64> = dm.row(local_idx)
                .iter()
                .cloned()
                .filter(|&d| d > 0.0) // Exclude distance to self
                .collect();

            if dists.is_empty() { // Should not happen if indices.len() > 1, but good practice
                cores[i] = f64::INFINITY; // Indicate no neighbors within the cluster
                continue;
            }

            // Avoid division by zero or issues with infinite distances
            let sum_inv: f64 = dists.iter()
                                   .map(|&d| if d.is_finite() && d > 0.0 { 1.0 / d } else { 0.0 }) // Handle potential inf/zero distance
                                   .map(|inv_d| inv_d.powi(dims as i32)) // Power after inversion
                                   .sum();

            let count = dists.len() as f64;
            if sum_inv > 0.0 && count > 0.0 {
                // Ensure the base of the power is non-negative
                 let base = sum_inv / count;
                 if base > 0.0 {
                    cores[i] = base.powf(-1.0 / dims as f64);
                 } else {
                    cores[i] = f64::INFINITY; // Or handle as error/specific case
                 }

            } else {
                cores[i] = f64::INFINITY; // Or 0.0, depends on algorithm definition for single/no-neighbor points
            }
        }
    }
    cores
}


/// Build mutual reachability graph matrix
fn mutual_reach_graph(x: &ArrayView2<f64>, cores: &[f64]) -> Array2<f64> {
    let n = x.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&x.row(i), &x.row(j));
            // Ensure core distances are non-negative before max
            let core_i = cores[i].max(0.0);
            let core_j = cores[j].max(0.0);
            let mr = core_i.max(core_j).max(d);
            out[(i, j)] = mr;
            out[(j, i)] = mr;
        }
    }
    out
}


/// Compute MST matrix using Prim's algorithm - returns Adjacency List representation for sparsity
/// Returns Vec<(usize, usize, f64)> representing edges (u, v, weight)
fn mutual_reach_mst_edges(graph: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let n = graph.shape()[0];
    if n == 0 {
        return Vec::new();
    }

    let mut in_mst = vec![false; n];
    let mut key = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n]; // usize::MAX indicates no parent
    let mut mst_edges = Vec::with_capacity(n - 1);

    key[0] = 0.0; // Start Prim's from node 0

    for _ in 0..n {
        // Find the vertex u not yet in MST with the minimum key value
        let mut min_key = f64::INFINITY;
        let mut u = usize::MAX;
        for v in 0..n {
            if !in_mst[v] && key[v] < min_key {
                min_key = key[v];
                u = v;
            }
        }

        // If u remains usize::MAX, it means the remaining vertices are unreachable (disconnected graph)
        if u == usize::MAX {
            // Optionally handle disconnected graphs here, e.g., start Prim's again from an unvisited node
            // For simplicity now, we break. This assumes a connected graph which MR-distance should ensure
            // if there are no completely isolated points with infinite core distances.
            break;
        }

        in_mst[u] = true;

        // If u has a parent (i.e., it's not the starting node 0 with key 0), add the edge to MST
        if parent[u] != usize::MAX {
            let p = parent[u];
            mst_edges.push((p, u, graph[(p, u)]));
        }

        // Update key values of adjacent vertices of u
        for v in 0..n {
            if !in_mst[v] {
                let weight = graph[(u, v)];
                if weight < key[v] && weight.is_finite() { // Check for finite weight
                    parent[v] = u;
                    key[v] = weight;
                }
            }
        }
    }
    mst_edges
}

// Helper to build an adjacency list representation of the MST from edges
// Vec<Vec<(usize, f64)>> where index `u` contains Vec of `(v, weight)`
fn build_mst_adj_list(n: usize, mst_edges: &[(usize, usize, f64)]) -> Vec<Vec<(usize, f64)>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v, w) in mst_edges {
         if w.is_finite() { // Only add edges with finite weights
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
    }
    adj
}


/// Overall clustering validity score using MST Adjacency List
fn clustering_validity(mst_edges: &[(usize, usize, f64)], labels: &ArrayView1<i32>) -> f64 {
    let n = labels.len();
    if n == 0 { return 0.0; } // Handle empty input

    // Identify unique cluster labels (excluding noise label if any, e.g., -1)
    let cluster_labels: Vec<i32> = labels.iter().cloned().filter(|&l| l >= 0).unique().collect();

    if cluster_labels.is_empty() { return 0.0; } // No valid clusters found

    // Build MST adjacency list once
    let mst_adj = build_mst_adj_list(n, mst_edges);

    let mut total_validity = 0.0;
    let mut total_weight = 0.0; // Use total points in valid clusters for weighting

    for &c in &cluster_labels {
        // Get indices of points belonging to the current cluster 'c'
        let indices: Vec<usize> = labels.iter()
                                        .enumerate()
                                        .filter_map(|(i, &l)| if l == c { Some(i) } else { None })
                                        .collect();

        if indices.len() > 0 { // Calculate validity only for non-empty clusters
             let cluster_weight = indices.len() as f64;
             let validity = single_cluster_validity(&mst_adj, &indices, labels);
             total_validity += validity * cluster_weight;
             total_weight += cluster_weight;
         }
    }

    if total_weight > 0.0 {
        total_validity / total_weight // Weighted average validity
    } else {
        0.0 // Return 0 if there are no points in valid clusters
    }
}


/// Validity of a single cluster using MST Adjacency List
fn single_cluster_validity(
    mst_adj: &Vec<Vec<(usize, f64)>>,
    cluster_indices: &[usize],
    all_labels: &ArrayView1<i32>
) -> f64 {
    if cluster_indices.len() <= 1 {
        // Define validity for single-point clusters. DBCV paper might define this.
        // Often considered perfectly dense (sparseness=0) and infinitely separated.
        // Returning 1.0 might be appropriate if separation is prioritized.
        // Returning 0.0 might be neutral. Let's choose 1.0 assuming good separation.
        return 1.0;
    }
    let mut max_intra_cluster_edge: f64 = 0.0;
    let mut min_inter_cluster_edge = f64::INFINITY; // Separation = min edge weight *connecting* to other clusters

    let cluster_indices_set: std::collections::HashSet<usize> = cluster_indices.iter().cloned().collect();

    for &u in cluster_indices {
        for &(v, weight) in &mst_adj[u] {
             if weight.is_finite() { // Consider only finite edges
                 if cluster_indices_set.contains(&v) {
                     // Edge (u, v) is within the same cluster
                     // Since MST edges are added undirected, only need to consider u < v to count once
                     if u < v {
                          max_intra_cluster_edge = max_intra_cluster_edge.max(weight);
                     }
                 } else {
                     // Edge (u, v) connects to a different cluster (or noise)
                     min_inter_cluster_edge = min_inter_cluster_edge.min(weight);
                 }
             }
         }
    }

     // Handle cases where separation might be infinite (cluster is completely isolated in the MST)
     // or sparseness is zero (e.g., all points coincident, though unlikely with MR-distance).
     let sparseness = max_intra_cluster_edge;
     let separation = min_inter_cluster_edge;

     if separation.is_infinite() && sparseness == 0.0 {
         // Isolated single point or coincident points, considered perfectly valid.
         return 1.0;
     }
     if separation.is_infinite() {
         // Finite sparseness but infinite separation (isolated cluster)
         // This indicates good separation. Depending on definition, could be 1.0.
         return 1.0;
     }
     if separation == 0.0 && sparseness == 0.0 {
        // Both zero - implies overlapping points considered as separate clusters?
        // Or potentially an issue in MST/core dist. Return 0.0 as neutral/undefined.
        return 0.0;
     }
     if separation.max(sparseness) == 0.0 { // Avoid division by zero if both somehow end up 0
         return 0.0;
     }


    // Original DBCV formula: (min_inter - max_intra) / max(min_inter, max_intra)
    (separation - sparseness) / separation.max(sparseness)

}

// Note: Removed the old MST matrix function and cluster validity functions
// that used the dense matrix representation, as the edge list/adjacency list
// approach is generally better for sparse MSTs.