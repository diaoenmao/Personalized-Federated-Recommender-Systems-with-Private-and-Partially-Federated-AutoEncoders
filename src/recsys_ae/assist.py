import copy
import torch
import models
import numpy as np
from scipy.sparse import csr_matrix
from config import cfg
from data import make_data_loader
from organization import Organization
from utils import make_optimizer, to_device


class Assist:
    def __init__(self, data_split):
        self.data_split = data_split
        self.num_organizations = len(data_split)
        self.model_name = self.make_model_name()
        self.ar_state_dict = [[None for _ in range(cfg['num_organizations'])] for _ in
                              range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{k: None for k in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{k: None for k in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        return

    def make_model_name(self):
        model_name = [cfg['model_name'] for _ in range(cfg['global']['num_epochs'] + 1)]
        model_name = [model_name for _ in range(self.num_organizations)]
        return model_name

    def make_organization(self):
        model_name = self.model_name
        organization = [None for _ in range(self.num_organizations)]
        for i in range(self.num_organizations):
            model_name_i = model_name[i]
            data_split_i = self.data_split[i]
            organization[i] = Organization(i, data_split_i, model_name_i)
        return organization

    def make_dataset(self, dataset, iter):
        for k in dataset[0]:
            output_k = torch.tensor(self.organization_output[iter - 1][k].data, dtype=torch.float32)
            target_k = torch.tensor(self.organization_target[0][k].data, dtype=torch.float32)
            output_k.requires_grad = True
            loss = models.loss_fn(output_k, target_k, reduction='sum')
            loss.backward()
            residual_k = - copy.deepcopy(output_k.grad)
            output_k.detach_()
            for i in range(len(dataset)):
                coo = self.organization_target[0][k].tocoo()
                row, col = coo.row, coo.col
                if hasattr(dataset[i][k], 'user_profile') and 'target' in dataset[i][k].user_profile:
                    del dataset[i][k].user_profile['target']
                if hasattr(dataset[i][k], 'item_attr') and 'target' in dataset[i][k].item_attr:
                    del dataset[i][k].item_attr['target']
                if cfg['data_mode'] == 'user':
                    dataset[i][k].target = csr_matrix((residual_k, (row, col)),
                                                      shape=(cfg['num_users']['target'], cfg['num_items']['target']))
                    dataset[i][k].transform.transforms[0].num_items['target'] = cfg['num_items']['target']
                elif cfg['data_mode'] == 'item':
                    dataset[i][k].target = csr_matrix((residual_k, (row, col)),
                                                      shape=(cfg['num_items']['target'], cfg['num_users']['target']))
                    dataset[i][k].transform.transforms[0].num_users['target'] = cfg['num_users']['target']
                else:
                    raise ValueError('Not valid data mode')

        return dataset

    def update(self, organization_outputs, iter):
        updated_data = {k: [None for i in range(len(organization_outputs))] for k in organization_outputs[0]}
        updated_row = {k: [None for i in range(len(organization_outputs))] for k in organization_outputs[0]}
        updated_col = {k: [None for i in range(len(organization_outputs))] for k in organization_outputs[0]}
        for i in range(len(organization_outputs)):
            for split in organization_outputs[0]:
                if split == 'train':
                    model = models.assist().to(cfg['device'])
                    if cfg['assist']['ar_mode'] == 'optim' or cfg['assist']['aw_mode'] == 'optim':
                        history = torch.tensor(self.organization_output[iter - 1][split][:, self.data_split[i]].data)
                        if 'match_rate' in cfg['assist']:
                            output = []
                            for j in range(len(organization_outputs)):
                                output_j = torch.tensor(organization_outputs[i][split][:, self.data_split[i]].data)
                                output_j_other = torch.tensor(
                                    organization_outputs[j][split][:, self.data_split[i]].data)
                                num_matched = int(len(output_j) * cfg['assist']['match_rate'])
                                output_j[:num_matched] = output_j_other[:num_matched]
                                output.append(output_j)
                        else:
                            output = [torch.tensor(organization_outputs[j][split][:, self.data_split[i]].data) for j in
                                      range(len(organization_outputs))]
                        output = torch.stack(output, dim=-1)
                        target = torch.tensor(self.organization_target[0][split][:, self.data_split[i]].data)
                        model.train(True)
                        input = {'history': history, 'output': output, 'target': target}
                        input = to_device(input, cfg['device'])
                        optimizer = make_optimizer(model, 'assist')
                        for _ in range(1, cfg['assist']['num_epochs'] + 1):
                            def closure():
                                output = model(input)
                                optimizer.zero_grad()
                                output['loss'].backward()
                                return output['loss']

                            optimizer.step(closure)
                    self.ar_state_dict[iter][i] = {k: v.cpu() for k, v in model.state_dict().items()}
                with torch.no_grad():
                    model = models.assist().to(cfg['device'])
                    model.load_state_dict(self.ar_state_dict[iter][i])
                    model.train(False)
                    history = torch.tensor(self.organization_output[iter - 1][split][:, self.data_split[i]].data)
                    if 'match_rate' in cfg['assist']:
                        output = []
                        for j in range(len(organization_outputs)):
                            output_j = torch.tensor(organization_outputs[i][split][:, self.data_split[i]].data)
                            output_j_other = torch.tensor(
                                organization_outputs[j][split][:, self.data_split[i]].data)
                            num_matched = int(len(output_j) * cfg['assist']['match_rate'])
                            output_j[:num_matched] = output_j_other[:num_matched]
                            output.append(output_j)
                    else:
                        output = [torch.tensor(organization_outputs[j][split][:, self.data_split[i]].data) for j in
                                  range(len(organization_outputs))]
                    output = torch.stack(output, dim=-1)
                    target = torch.tensor(self.organization_target[0][split][:, self.data_split[i]].data)
                    input = {'history': history, 'output': output, 'target': target}
                    input = to_device(input, cfg['device'])
                    output = model(input)
                    updated_data[split][i] = output['target'].cpu().numpy()
                    coo = self.organization_output[iter - 1][split][:, self.data_split[i]].tocoo()
                    updated_row[split][i] = coo.row
                    updated_col[split][i] = self.data_split[i][coo.col].cpu().numpy()
        for k in organization_outputs[0]:
            updated_data_k = np.concatenate(updated_data[k])
            updated_row_k = np.concatenate(updated_row[k])
            updated_col_k = np.concatenate(updated_col[k])
            if cfg['data_mode'] == 'user':
                self.organization_output[iter][k] = csr_matrix((updated_data_k, (updated_row_k, updated_col_k)),
                                                               shape=(cfg['num_users']['target'],
                                                                      cfg['num_items']['target']))
            elif cfg['data_mode'] == 'item':
                self.organization_output[iter][k] = csr_matrix((updated_data_k, (updated_row_k, updated_col_k)),
                                                               shape=(cfg['num_items']['target'],
                                                                      cfg['num_users']['target']))
            else:
                raise ValueError('Not valid data mode')
        return
