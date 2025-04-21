"""
Adapted from:
-------------------------------------------------------------------------------
Copyright (c) 2021, QUVA-Lab, University of Amsterdam
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""

import json
import os
from datetime import datetime
import sys
from argparse import ArgumentParser
sys.path.append("../")

from data.graph_visualization import visualize_graph
from data.graph_generation import generate_categorical_graph, get_graph_func
from data.utils import set_seed


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--graph_type', type=str, default='random',
                        help='Which graph type to test on. Currently supported are: '
                             'chain, bidiag, collider, jungle, full, regular, random, '
                             'random_max_#N where #N is to be replaced with an integer. '
                             'random_max_10 is random with max. 10 parents per node.')
    parser.add_argument('--num_graphs', type=int, default=1,
                        help='Number of graphs to generate and sequentially test on.')
    parser.add_argument('--num_vars', type=int, default=8,
                        help='Number of variables that the graphs should have.')
    parser.add_argument('--num_categs', type=int, default=10,
                        help='Number of categories/different values each variable can take.')
    parser.add_argument('--edge_prob', type=float, default=0.2,
                        help='For random graphs, the probability of two arbitrary nodes to be connected.')
    args = parser.parse_args()

    for num_vars in args.num_vars:
        for num_categs in args.num_categs:
            for edge_prob in args.edge_prob:
                # TODO: replace all args.blablabla
                # TODO: generate 100 samples per interv, and 1000 observational
                # Basic checkpoint directory creation
                folder = f"datasets/{args.graph_type}_{args.num_vars}_{args.num_categs}_{args.edge_prob}"
                os.makedirs(folder, exist_ok=True)

                with open(os.path.join(folder, "args.json"), "w") as f:
                    json.dump(vars(args), f, indent=4)

                for gindex in range(args.num_graphs):
                    # Seed setting for reproducibility
                    set_seed(args.seed+gindex)  # Need to increase seed, otherwise we might same graphs
                    # Generate graph
                    print("Generating %s graph with %i variables..." % (args.graph_type, args.num_vars))
                    graph = generate_categorical_graph(num_vars=args.num_vars,
                                                    min_categs=args.num_categs,
                                                    max_categs=args.num_categs,
                                                    edge_prob=args.edge_prob,
                                                    connected=True,
                                                    use_nn=True,
                                                    graph_func=get_graph_func(args.graph_type),
                                                    seed=args.seed+gindex)
                    file_id = "%s_%s" % (str(gindex+1).zfill(3), args.graph_type)
                    # Save graph
                    graph.save_to_file(os.path.join(folder, "graph_%s.pt" % (file_id)))
                    # Visualize graph
                    if graph.num_vars <= 100:
                        print("Visualizing graph...")
                        figsize = max(3, graph.num_vars ** 0.7)
                        visualize_graph(graph,
                                        filename=os.path.join(folder, "graph_%s.pdf" % (file_id)),
                                        figsize=(figsize, figsize),
                                        layout="circular" if graph.num_vars < 40 else "graphviz")
