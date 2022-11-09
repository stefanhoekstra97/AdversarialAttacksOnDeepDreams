import json
import run_lighting

import foolbox as fb



with open('hparams_models.json') as f:
    models_to_train = json.load(f)

vnum_start = 3
vnum_stop = 5

folder_prefix = "batchnormexpMNIST/"

attack_vectors = {
    # "FGSM-NORS": fb.attacks.FGSM(random_start=False), 
    # "FGSM-RS": fb.attacks.FGSM(random_start=True), 
    # "PGD5-0.4": fb.attacks.LinfPGD(steps=5, rel_stepsize=0.4), 
    # "PGD10-0.2": fb.attacks.LinfPGD(steps=10, rel_stepsize=0.2),
    # "PGD20-0.1": fb.attacks.LinfPGD(steps=20, rel_stepsize=0.1),
    # "CW10-0.2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=10, stepsize = 0.1, binary_search_steps=1, initial_const=1e-1),
    "CW20-0.1": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=20, stepsize = 0.1, binary_search_steps=1, initial_const=0.1, abort_early=True),
    # "CW20-0.1-bs3-conf2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=20, stepsize = 0.1, binary_search_steps=3, confidence=2, initial_const=1e-1),
    # "CW20-0.1-bs3-ic1e-2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=20, stepsize = 0.1, binary_search_steps=3, initial_const=1e-2),
    # "CW40-0.1-bs5-ic1e-3": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=40, stepsize = 0.1, binary_search_steps=5, initial_const=1e-3),
}

seed_per_vnum = [52, 53, 54, 55, 56, 57, 58]
skip_num = 0

_trainer_arg = {
    "deterministic": False,
    "num_sanity_val_steps": 0,
    "log_every_n_steps": 10
}

model_num = 0

for model_idx, (model_identifier, modelparams) in enumerate(models_to_train.items()):
    if "dream_params" in modelparams:
        dream_params = modelparams["dream_params"]
    else:
        dream_params = {}
    model_arch = modelparams["model_params"]["nn_arch"]

    if modelparams["model_params"]["with_adversarial"]:
        _attack_vector_list = attack_vectors
    else:
        _attack_vector_list = {"clean_training" : None}
    
    for attack_id, attack_v in _attack_vector_list.items():
        if "CW" in attack_id:
            attack_eps  = modelparams["model_params"]["l2_eps"]
        else:
            attack_eps  = modelparams["model_params"]["li_eps"]

        if "clean_training" in attack_id:
            run_identifier = model_arch + "/"  + model_identifier
        else:
            run_identifier = model_arch + "/" + attack_id + "/eps=" + str(attack_eps) + "/" + model_identifier

        for vnum in range(vnum_stop - vnum_start):
            vnum = vnum + vnum_start
            if model_num < skip_num:
                model_num += 1
                continue

            model_experiment_params = dict(modelparams)
            dream_experiment_params = dict(dream_params)
            v_hparams = {
                **model_experiment_params["hparams"],
                "model_identifier": folder_prefix + run_identifier,
                "version_nr": vnum,
                "seed": seed_per_vnum[vnum]
            }

            v_modelparams = dict(model_experiment_params["model_params"])

            if "CW" in attack_id:
                v_modelparams["adv_eps"] = v_modelparams["l2_eps"]
                del v_modelparams["l2_eps"], v_modelparams["li_eps"]
            else:
                v_modelparams["adv_eps"] = v_modelparams["li_eps"]
                del v_modelparams["l2_eps"], v_modelparams["li_eps"]

            run_lighting.main(hparams=v_hparams, trainer_kwargs=_trainer_arg, model_kwargs=v_modelparams, dream_kwargs=dream_experiment_params, attack=attack_v)
            if dream_params and vnum == 0:
                with open("trained_models/" + folder_prefix + run_identifier + "/dream_params.json", "w") as f:
                    json.dump(dream_params, f, indent=4)
                    f.close()



