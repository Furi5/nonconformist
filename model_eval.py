import chemprop

arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    '--checkpoint_dir', 'test_checkpoints_reg'
]

args = chemprop.args.PredictArgs().parse_args(arguments)
model_objects = chemprop.train.load_model(args=args)
smiles = [['CCC'], ['CCCC'], ['OCC']]

preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)