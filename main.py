import tensorflow as tf
from model import create_model
from data import prepare_dataset
from trainer import train_model, evaluate_model

def setup_tpu_or_gpu():
    """Setup TPU if available, otherwise return default strategy."""
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        print("Running on TPU")
        return strategy
    except Exception as e:
        print(f"No TPU detected. Error: {e}")
    
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
        print(f"Running on {len(strategy.extended.worker_devices)} GPU(s)")
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")
    
    return strategy

def main():
    BATCH_SIZE = 512
    NUM_CLASSES = 10
    EPOCHS = 300
    INITIAL_LR = 0.01
    
    strategy = setup_tpu_or_gpu()
    train_ds, test_ds, train_size = prepare_dataset(
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )
    

    with strategy.scope():
        model = create_model(
            image_size=32,
            projection_dim=128,
            depth=8,
            patch_size=2,
            num_classes=NUM_CLASSES
        )
    
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR
    )

    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_ds)

    max_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {max_val_acc:.4f}")
    
    return model, history, metrics

if __name__ == "__main__":
    main()