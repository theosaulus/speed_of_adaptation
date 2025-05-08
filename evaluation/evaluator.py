import numpy as np
import copy
import torch
import torch.nn.functional as F

from objectives.pseudo_ll import pseudo_ll_loss
from data.dataset_creation import sample_dict_to_tensor
from evaluation.compute_nll import compute_nll_bound, compute_nll_on_ground_truth

def evaluate_bounds(graph, dataset, order, device):
    results = {}

    name2gidx = {v.name: i for i, v in enumerate(graph.variables)}
    name2xcol = {name: idx for idx, name in enumerate(order)}
    
    node_relations, one_hop_relations, global_roles, in_degrees, out_degrees, total_degrees =\
        graph.node_relations
    
    # Observational regime
    if 'observational' in dataset:
        X_obs, _ = sample_dict_to_tensor(dataset['observational'], device, order)
        X_obs = X_obs.to(device)
        results['bound_obs_nointerv'] = compute_nll_bound(graph, graph, X_obs, order)

    # Interventional regimes
    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, device, order)
        X_int = X_int.to(device)

        # var_index = order.index(var_name)
        # constant = int(X_int[0, var_index].item())
        # graph_int = graph.get_intervened_graph({var_name: constant})

        # num_vars = X_int.shape[1]
        # all_idxs = list(range(num_vars))
        x_col = name2xcol[var_name]
        gidx  = name2gidx[var_name]

        # 3) Create the intervened graph correctly
        constant  = int(X_int[0, x_col].item())
        graph_int = graph.get_intervened_graph({var_name: constant})

        # subsets wrt this intervention target
        parents = np.where(one_hop_relations[gidx] == 1)[0].tolist()
        children = np.where(one_hop_relations[gidx] == -1)[0].tolist()
        ancestors = np.where(node_relations[gidx] == 1)[0].tolist()
        descendants = np.where(node_relations[gidx] == -1)[0].tolist()
        roots = np.where(global_roles == 'root')[0].tolist()
        leaves = np.where(global_roles == 'leaf')[0].tolist()

        bound_obs_full = compute_nll_bound(graph, graph_int, X_int, order)
        bound_obs_intervention = compute_nll_bound(graph, graph_int, X_int, order, var_indices=[gidx])
        bound_obs_root = compute_nll_bound(graph, graph_int, X_int, order, var_indices=roots) if roots else None
        bound_obs_leaf = compute_nll_bound(graph, graph_int, X_int, order, var_indices=leaves) if leaves else None
        bound_obs_ancestor = compute_nll_bound(graph, graph_int, X_int, order, var_indices=ancestors) if ancestors else None
        bound_obs_descendant = compute_nll_bound(graph, graph_int, X_int, order, var_indices=descendants) if descendants else None
        bound_obs_parent = compute_nll_bound(graph, graph_int, X_int, order, var_indices=parents) if parents else None
        bound_obs_child = compute_nll_bound(graph, graph_int, X_int, order, var_indices=children) if children else None

        bound_int_full = compute_nll_bound(graph_int, graph_int, X_int, order)
        bound_int_intervention = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=[gidx])
        bound_int_root = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=roots) if roots else None
        bound_int_leaf = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=leaves) if leaves else None
        bound_int_ancestor = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=ancestors) if ancestors else None
        bound_int_descendant = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=descendants) if descendants else None
        bound_int_parent = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=parents) if parents else None
        bound_int_child = compute_nll_bound(graph_int, graph_int, X_int, order, var_indices=children) if children else None

        results[f'bound_obs_{var_name}_full'] = bound_obs_full
        results[f'bound_obs_{var_name}_intervention'] = bound_obs_intervention
        results[f'bound_obs_{var_name}_root'] = bound_obs_root
        results[f'bound_obs_{var_name}_leaf'] = bound_obs_leaf
        results[f'bound_obs_{var_name}_ancestor'] = bound_obs_ancestor
        results[f'bound_obs_{var_name}_descendant'] = bound_obs_descendant
        results[f'bound_obs_{var_name}_parent'] = bound_obs_parent
        results[f'bound_obs_{var_name}_child'] = bound_obs_child

        results[f'bound_int_{var_name}_full'] = bound_int_full
        results[f'bound_int_{var_name}_intervention'] = bound_int_intervention
        results[f'bound_int_{var_name}_root'] = bound_int_root
        results[f'bound_int_{var_name}_leaf'] = bound_int_leaf
        results[f'bound_int_{var_name}_ancestor'] = bound_int_ancestor
        results[f'bound_int_{var_name}_descendant'] = bound_int_descendant
        results[f'bound_int_{var_name}_parent'] = bound_int_parent
        results[f'bound_int_{var_name}_child'] = bound_int_child

        inter_vars = list(dataset.get('interventional', {}).keys())
        for metric in ('bound_obs', 'bound_int'):
            for subset in ('full', 'intervention', 'root', 'leaf', 'ancestor', 'descendant', 'parent', 'child'):
                key = f"{metric}_all_{subset}"
                vals = []
                for v in inter_vars:
                    k = f"{metric}_{v}_{subset}"
                    if k in results and results[k] is not None:
                        vals.append(results[k])
                if vals:
                    results[key] = np.mean([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in vals])

    return results    

def evaluate_zero_shot(model, graph, dataset, order, device):
    """
    Zero-shot evaluation across all regimes (observational + each intervention).
    Returns a dict with per‑regime NLL.
    """
    results = {}
    node_relations, one_hop_relations, global_roles, in_degrees, out_degrees, total_degrees =\
        graph.node_relations

    # Observational regime
    if 'observational' in dataset:
        X_obs, _ = sample_dict_to_tensor(dataset['observational'], device, order)
        X_obs = X_obs.to(device)
        with torch.no_grad():
            results['raw_pseudo_nll_obs'] = - pseudo_ll_loss(model, X_obs)
            results['nll_on_gt_obs'] = compute_nll_on_ground_truth(model, graph, X_obs)

    # Interventional regimes
    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, device, order)
        X_int = X_int.to(device)

        var_index = order.index(var_name)
        graph_int = graph.get_intervened_graph({var_name: X_int[:, var_index]})

        with torch.no_grad():
            num_vars = X_int.shape[1]
            all_idxs = list(range(num_vars))

            # subsets wrt this intervention target
            parents = np.where(one_hop_relations[var_index] == 1)[0].tolist()
            children = np.where(one_hop_relations[var_index] == -1)[0].tolist()
            ancestors = np.where(node_relations[var_index] == 1)[0].tolist()
            descendants = np.where(node_relations[var_index] == -1)[0].tolist()
            roots = np.where(global_roles == 'root')[0].tolist()
            leaves = np.where(global_roles == 'leaf')[0].tolist()

            # raw pseudo-LL on each subset
            raw_nll_full        = pseudo_ll_loss(model, X_int)
            raw_nll_intervention= pseudo_ll_loss(model, X_int, var_indices=[var_index])
            raw_nll_root        = pseudo_ll_loss(model, X_int, var_indices=roots)      if roots        else None
            raw_nll_leaf        = pseudo_ll_loss(model, X_int, var_indices=leaves)     if leaves       else None
            raw_nll_ancestor    = pseudo_ll_loss(model, X_int, var_indices=ancestors)  if ancestors    else None
            raw_nll_descendant  = pseudo_ll_loss(model, X_int, var_indices=descendants)if descendants  else None
            raw_nll_parent      = pseudo_ll_loss(model, X_int, var_indices=parents)    if parents      else None
            raw_nll_child       = pseudo_ll_loss(model, X_int, var_indices=children)   if children     else None

            # ground-truth NLL on each subset (observational graph)
            nll_obs_full        = compute_nll_on_ground_truth(model, graph, X_int, order)
            nll_obs_intervention= compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=[var_index])
            nll_obs_root        = compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=roots)       if roots        else None
            nll_obs_leaf        = compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=leaves)      if leaves       else None
            nll_obs_ancestor    = compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=ancestors)   if ancestors    else None
            nll_obs_descendant  = compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=descendants) if descendants  else None
            nll_obs_parent      = compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=parents)     if parents      else None
            nll_obs_child       = compute_nll_on_ground_truth(model, graph, X_int, order, var_indices=children)    if children     else None

            # ground-truth NLL on each subset (intervened graph)
            nll_int_full        = compute_nll_on_ground_truth(model, graph_int, X_int, order)
            nll_int_intervention= compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=[var_index])
            nll_int_root        = compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=roots)       if roots        else None
            nll_int_leaf        = compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=leaves)      if leaves       else None
            nll_int_ancestor    = compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=ancestors)   if ancestors    else None
            nll_int_descendant  = compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=descendants) if descendants  else None
            nll_int_parent      = compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=parents)     if parents      else None
            nll_int_child       = compute_nll_on_ground_truth(model, graph_int, X_int, order, var_indices=children)    if children     else None

        def _to_numpy(tensor):
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            return tensor
            
        results[f'raw_pseudo_nll_{var_name}_full']         = _to_numpy(raw_nll_full)
        results[f'raw_pseudo_nll_{var_name}_intervention'] = _to_numpy(raw_nll_intervention)
        results[f'raw_pseudo_nll_{var_name}_root']         = _to_numpy(raw_nll_root)
        results[f'raw_pseudo_nll_{var_name}_leaf']         = _to_numpy(raw_nll_leaf)
        results[f'raw_pseudo_nll_{var_name}_ancestor']     = _to_numpy(raw_nll_ancestor)
        results[f'raw_pseudo_nll_{var_name}_descendant']   = _to_numpy(raw_nll_descendant)
        results[f'raw_pseudo_nll_{var_name}_parent']       = _to_numpy(raw_nll_parent)
        results[f'raw_pseudo_nll_{var_name}_child']        = _to_numpy(raw_nll_child)

        results[f'nll_on_gt_obs_{var_name}_full']         = _to_numpy(nll_obs_full)
        results[f'nll_on_gt_obs_{var_name}_intervention'] = _to_numpy(nll_obs_intervention)
        results[f'nll_on_gt_obs_{var_name}_root']         = _to_numpy(nll_obs_root)
        results[f'nll_on_gt_obs_{var_name}_leaf']         = _to_numpy(nll_obs_leaf)
        results[f'nll_on_gt_obs_{var_name}_ancestor']     = _to_numpy(nll_obs_ancestor)
        results[f'nll_on_gt_obs_{var_name}_descendant']   = _to_numpy(nll_obs_descendant)
        results[f'nll_on_gt_obs_{var_name}_parent']       = _to_numpy(nll_obs_parent)
        results[f'nll_on_gt_obs_{var_name}_child']        = _to_numpy(nll_obs_child)

        results[f'nll_on_gt_int_{var_name}_full']         = _to_numpy(nll_int_full)
        results[f'nll_on_gt_int_{var_name}_intervention'] = _to_numpy(nll_int_intervention)
        results[f'nll_on_gt_int_{var_name}_root']         = _to_numpy(nll_int_root)
        results[f'nll_on_gt_int_{var_name}_leaf']         = _to_numpy(nll_int_leaf)
        results[f'nll_on_gt_int_{var_name}_ancestor']     = _to_numpy(nll_int_ancestor)
        results[f'nll_on_gt_int_{var_name}_descendant']   = _to_numpy(nll_int_descendant)
        results[f'nll_on_gt_int_{var_name}_parent']       = _to_numpy(nll_int_parent)
        results[f'nll_on_gt_int_{var_name}_child']        = _to_numpy(nll_int_child)

    # Compute averages over the different interventions
    inter_vars = list(dataset.get('interventional', {}).keys())

    # for each metric type and each subset suffix
    for metric in ('raw_pseudo_nll', 'nll_on_gt_obs', 'nll_on_gt_int'):
        for subset in ('full', 'intervention', 'root', 'leaf', 'ancestor', 'descendant', 'parent', 'child'):
            key = f"{metric}_all_{subset}"
            vals = []
            for v in inter_vars:
                k = f"{metric}_{v}_{subset}"
                if k in results and results[k] is not None:
                    vals.append(results[k])
            if vals:
                results[key] = np.mean([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in vals])

    return results

def evaluate_few_shot(
        model, 
        graph, 
        dataset, 
        order, 
        few_shot_num_samples, 
        few_shot_gradient_steps, 
        device,
        few_shot_lr=0.1,
    ):
    """
    Few-shot evaluation across all interventional regimes.
    Returns a dict with per‑regime NLL, for the list of shots and the given 
    number of gradient steps.
    """
    S = few_shot_gradient_steps
    results = {}
    node_relations, one_hop_relations, global_roles, in_degrees, out_degrees, total_degrees =\
        graph.node_relations

    for K in few_shot_num_samples:
        for var_name, sample_dict in dataset['interventional'].items():
            X_int, _ = sample_dict_to_tensor(sample_dict, device, order)
            X_int = X_int.to(device)
            N = X_int.size(0)

            var_index = order.index(var_name)
            graph_int = graph.get_intervened_graph({var_name: X_int[:, var_index]})

            # if not enough samples error
            if K > N:
                print(f"Not enough samples for {var_name} regime: {N} < {K}")
                continue
                # raise ValueError(f"Not enough samples for {var_name} regime: {N} < {K}")
            
            else:
                finetuned_model = copy.deepcopy(model).to(device).train()
                # TODO: allow for sparse adaptation
                # for param in finetuned_model.parameters(): 
                #     param.requires_grad = True
                optimizer = torch.optim.SGD(finetuned_model.parameters(), lr=few_shot_lr)
                idx = np.random.choice(N, K, replace=False)
                X_adapt = X_int[idx]

                for _ in range(few_shot_gradient_steps):
                    loss_i = pseudo_ll_loss(finetuned_model, X_adapt)
                    optimizer.zero_grad()
                    loss_i.backward()
                    optimizer.step()
                
                finetuned_model.eval()

            with torch.no_grad():
                num_vars = X_int.shape[1]
                all_idxs = list(range(num_vars))

                # subsets wrt this intervention target
                parents = np.where(one_hop_relations[var_index] == 1)[0].tolist()
                children = np.where(one_hop_relations[var_index] == -1)[0].tolist()
                ancestors = np.where(node_relations[var_index] == 1)[0].tolist()
                descendants = np.where(node_relations[var_index] == -1)[0].tolist()
                roots = np.where(global_roles == 'root')[0].tolist()
                leaves = np.where(global_roles == 'leaf')[0].tolist()

                # raw pseudo-LL on each subset
                raw_nll_full        = pseudo_ll_loss(finetuned_model, X_int)
                raw_nll_intervention= pseudo_ll_loss(finetuned_model, X_int, var_indices=[var_index])
                raw_nll_root        = pseudo_ll_loss(finetuned_model, X_int, var_indices=roots)      if roots        else None
                raw_nll_leaf        = pseudo_ll_loss(finetuned_model, X_int, var_indices=leaves)     if leaves       else None
                raw_nll_ancestor    = pseudo_ll_loss(finetuned_model, X_int, var_indices=ancestors)  if ancestors    else None
                raw_nll_descendant  = pseudo_ll_loss(finetuned_model, X_int, var_indices=descendants)if descendants  else None
                raw_nll_parent      = pseudo_ll_loss(finetuned_model, X_int, var_indices=parents)    if parents      else None
                raw_nll_child       = pseudo_ll_loss(finetuned_model, X_int, var_indices=children)   if children     else None

                # ground-truth NLL on each subset (observational graph)
                nll_obs_full        = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order)
                nll_obs_intervention= compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=[var_index])
                nll_obs_root        = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=roots)       if roots        else None
                nll_obs_leaf        = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=leaves)      if leaves       else None
                nll_obs_ancestor    = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=ancestors)   if ancestors    else None
                nll_obs_descendant  = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=descendants) if descendants  else None
                nll_obs_parent      = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=parents)     if parents      else None
                nll_obs_child       = compute_nll_on_ground_truth(finetuned_model, graph, X_int, order, var_indices=children)    if children     else None

                # ground-truth NLL on each subset (intervened graph)
                nll_int_full        = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order)
                nll_int_intervention= compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=[var_index])
                nll_int_root        = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=roots)       if roots        else None
                nll_int_leaf        = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=leaves)      if leaves       else None
                nll_int_ancestor    = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=ancestors)   if ancestors    else None
                nll_int_descendant  = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=descendants) if descendants  else None
                nll_int_parent      = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=parents)     if parents      else None
                nll_int_child       = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int, order, var_indices=children)    if children     else None

            def _to_numpy(tensor):
                if tensor is None:
                    return None
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy()
                return tensor
            
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_full']         = _to_numpy(raw_nll_full)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_intervention'] = _to_numpy(raw_nll_intervention)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_root']         = _to_numpy(raw_nll_root)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_leaf']         = _to_numpy(raw_nll_leaf)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_ancestor']     = _to_numpy(raw_nll_ancestor)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_descendant']   = _to_numpy(raw_nll_descendant)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_parent']       = _to_numpy(raw_nll_parent)
            results[f'raw_pseudo_nll_{var_name}_{S}_shot_{K}_ex_child']        = _to_numpy(raw_nll_child)

            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_full']         = _to_numpy(nll_obs_full)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_intervention'] = _to_numpy(nll_obs_intervention)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_root']         = _to_numpy(nll_obs_root)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_leaf']         = _to_numpy(nll_obs_leaf)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_ancestor']     = _to_numpy(nll_obs_ancestor)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_descendant']   = _to_numpy(nll_obs_descendant)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_parent']       = _to_numpy(nll_obs_parent)
            results[f'nll_on_gt_obs_{var_name}_{S}_shot_{K}_ex_child']        = _to_numpy(nll_obs_child)

            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_full']         = _to_numpy(nll_int_full)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_intervention'] = _to_numpy(nll_int_intervention)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_root']         = _to_numpy(nll_int_root)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_leaf']         = _to_numpy(nll_int_leaf)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_ancestor']     = _to_numpy(nll_int_ancestor)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_descendant']   = _to_numpy(nll_int_descendant)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_parent']       = _to_numpy(nll_int_parent)
            results[f'nll_on_gt_int_{var_name}_{S}_shot_{K}_ex_child']        = _to_numpy(nll_int_child)

        # Compute averages over the different interventions
        inter_vars = list(dataset.get('interventional', {}).keys())

        # for each metric type and each subset suffix
        for metric in ('raw_pseudo_nll', 'nll_on_gt_obs', 'nll_on_gt_int'):
            for subset in ('full', 'intervention', 'root', 'leaf', 'ancestor', 'descendant', 'parent', 'child'):
                key = f"{metric}_all_{S}_shot_{K}_ex_{subset}"
                vals = []
                for v in inter_vars:
                    k = f"{metric}_{v}_{S}_shot_{K}_ex_{subset}"
                    if k in results and results[k] is not None:
                        vals.append(results[k])
                if vals:
                    results[key] = np.mean([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in vals])

    return results