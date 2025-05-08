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
import numpy as np
import torch
sys.path.append("../")

from data.graph_visualization import visualize_graph
from data.graph_generation import generate_categorical_graph, get_graph_func
from data.utils import set_seed
from data.dataset_creation import build_dataset, sample_dict_to_tensor
from data.graph_export import export_graph

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--graph_type', type=str, default='random',
                        help='Which graph type to test on. Currently supported are: '
                             'chain, bidiag, collider, jungle, full, regular, random, '
                             'random_max_#N where #N is to be replaced with an integer. '
                             'random_max_10 is random with max. 10 parents per node.')
    parser.add_argument('--num_graphs', type=int, default=10,
                        help='Number of graphs to generate and sequentially test on.')
    parser.add_argument('--num_obs', type=int, default=2000,
                        help='Number of observational samples to generate.')
    parser.add_argument('--num_int', type=int, default=100,
                        help='Number of interventional samples per variable to generate.')
    parser.add_argument('--num_vars', type=int, default=[10, 20],
                        help='Number of variables that the graphs should have.')
    parser.add_argument('--num_categs', type=int, default=[10],
                        help='Number of categories/different values each variable can take.')
    parser.add_argument('--edge_prob', type=float, default=[0.3, 0.7],
                        help='For random graphs, the probability of two arbitrary nodes to be connected.')
    parser.add_argument('--connected_graphs', action='store_true', default=True,
                        help='If set, the generated graphs will be connected.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    for num_vars in args.num_vars:
        for num_categs in args.num_categs:
            for edge_prob in args.edge_prob:
                folder = f"datasets/{args.graph_type}_d{num_vars}_k{num_categs}_p{str(edge_prob).replace('.', '')}"
                os.makedirs(folder, exist_ok=True)

                with open(os.path.join(folder, "args.json"), "w") as f:
                    json.dump(vars(args), f, indent=4)

                for gindex in range(args.num_graphs):
                    seed = args.seed + gindex
                    set_seed(seed)  # Need to increase seed, otherwise we might same graphs
                    
                    # Generate graph
                    print("Generating %s graph with %i variables..." % (args.graph_type, num_vars))
                    graph = generate_categorical_graph(num_vars=num_vars,
                                                    min_categs=num_categs,
                                                    max_categs=num_categs,
                                                    edge_prob=edge_prob,
                                                    connected=args.connected_graphs,
                                                    use_nn=True,
                                                    graph_func=get_graph_func(args.graph_type),
                                                    seed=seed)
                    file_id = "%s_%s_%i_%i_%s" % (str(gindex+1).zfill(3), args.graph_type, num_vars, seed, "connected" if args.connected_graphs else "disconnected")
                    graph.save_to_file(os.path.join(folder, file_id + ".pt"))
                    export_graph(
                        filename=os.path.join(folder, file_id),
                        graph=graph,
                        num_obs=args.num_obs,
                        num_int=args.num_int,
                    )
                    print("Done for graph %s" % (file_id))

                    