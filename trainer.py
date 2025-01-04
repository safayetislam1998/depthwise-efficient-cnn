import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class OptimizerWeightDecay(tf.keras.callbacks.Callback):
    """Callback to update weight decay based on learning rate changes."""
    def __init__(self, lr_base, wd_base):
        super(OptimizerWeightDecay, self).__init__()
        self.wd_m = wd_base / lr_base
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.model is not None:
            lr = K.get_value(self.model.optimizer.lr)
            wd = self.wd_m * lr
            K.set_value(self.model.optimizer.weight_decay, wd)
            print(f'Epoch {epoch}: weight decay set to {wd}')

def get_lr_schedule(initial_lr=0.01, min_lr=5e-5):
    """Create learning rate schedule with warmup and cosine decay."""
    def lr_schedule(epoch):
        # Warm-up phase (first 5 epochs)
        if epoch < 5:
            return (initial_lr - min_lr) / 5 * epoch + min_lr
            
        # After warmup, use step decay
        decay_rate = 0.9
        decay_steps = 2
        return max(initial_lr * decay_rate ** ((epoch - 5) // decay_steps), min_lr)
        
    return lr_schedule

def train_model(model, train_ds, test_ds, epochs=300, initial_lr=0.01):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=initial_lr,
        weight_decay=initial_lr * 0.1  
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(
            get_lr_schedule(initial_lr), 
            verbose=1
        ),
        OptimizerWeightDecay(
            lr_base=initial_lr,
            wd_base=initial_lr * 0.1
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        workers=8,
        use_multiprocessing=True
    )
    
    return history

def evaluate_model(model, test_ds):
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision
    }