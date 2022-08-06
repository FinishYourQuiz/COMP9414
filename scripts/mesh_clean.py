import numpy as np
from tqdm import tqdm

def get_connected_component(mesh, arg_index=1):
  mesh_pred = mesh.copy()
  class ufds:
    parent_node = {}
    rank = {}

    def make_set(self, u):
      for i in u:
        self.parent_node[i] = i
        self.rank[i] = 0

    def op_find(self, k):
      if self.parent_node[k] != k:
        self.parent_node[k] = self.op_find(self.parent_node[k])
      return self.parent_node[k]

    def op_union(self, a, b):
      x = self.op_find(a)
      y = self.op_find(b)

      if x == y:
        return
      if self.rank[x] > self.rank[y]:
        self.parent_node[y] = x
      elif self.rank[x] < self.rank[y]:
        self.parent_node[x] = y
      else:
        self.parent_node[x] = y
        self.rank[y] = self.rank[y] + 1

  u = np.array(range(mesh_pred.vertices.shape[0]))
  data = ufds()
  data.make_set(u)

  for i in tqdm(range(mesh_pred.vertices.shape[0])):
    adjecent_verts = mesh_pred.vertex_neighbors[i]
    for j in adjecent_verts:
      data.op_union(i, j)

  cols = np.array([data.op_find(i) for i in range(mesh_pred.vertices.shape[0])])
  colors = np.unique(cols)

  counts = np.zeros(colors.shape)
  for c in range(colors.shape[0]):
    counts[c] = np.count_nonzero(cols == colors[c])
  max_n_args = np.argsort(counts)


  max_n_i = max_n_args[-arg_index]

  max_c = colors[max_n_i]
  # mesh_pred.vertices = mesh_pred.vertices[cols != max_c]
  mesh_pred.update_vertices(cols == max_c)
  # mesh_pred.update_faces(mesh_pred.remove_degenerate_faces())
  mesh_pred = mesh_pred.process(validate=True)
  mesh_pred.invert()

  return mesh_pred