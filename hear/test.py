import torch

from hear_mae.sample import load_model, get_scene_embeddings, get_timestamp_embeddings

if __name__ == '__main__':
    model = load_model(mode="logits").cuda()
    seconds = 15
    audio = torch.ones((3, 16000 * seconds))*0.5
    embed, time_stamps = get_timestamp_embeddings(audio, model)
    print(embed.shape)
    # print(time_stamps)
    embed = get_scene_embeddings(audio, model)
    print(embed.shape)
    print(embed[0, 10])
