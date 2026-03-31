import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# encoder = YourTrainedEncoder()
# generator_g = YourTrainedVectorFieldMLP() # The g(z) that preserves logits

class SymmetryAwareClassifier(nn.Module):
    def __init__(self, latent_dim=2, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, z):
        x = F.relu(self.fc1(z))
        return self.fc2(x)

def train_invariant_network(encoder, generator_g, dataloader, epochs=20, epsilon=0.1, reg_lambda=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = SymmetryAwareClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Freeze encoder and generator (we only train the classifier now)
    encoder.eval()
    generator_g.eval()
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # 1. Get Latent representation
                z, _ = encoder(images) # Assuming VAE returns z, mu, logvar
                
                # 2. Get the directional flow field
                flow_vector = generator_g(z)
                
                # 3. Create infinitesimally rotated latent vectors
                z_rotated = z + (epsilon * flow_vector)
            
            # 4. Standard Classification Loss
            logits_original = classifier(z)
            loss_cls = criterion(logits_original, labels)
            
            # 5. SYMMETRY REGULARIZATION LOSS (The Core Idea)
            # We want the classifier output to be identical for z and z_rotated
            logits_rotated = classifier(z_rotated)
            
            # Use Mean Squared Error to enforce invariance along the vector field
            loss_inv = F.mse_loss(logits_rotated, logits_original)
            
            # Total Loss
            loss = loss_cls + (reg_lambda * loss_inv)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {total_loss/len(dataloader):.4f}")
        
    return classifier

# To run it:
# invariant_clf = train_invariant_network(encoder, g_net, train_loader)
