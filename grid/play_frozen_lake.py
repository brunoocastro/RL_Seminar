"""
Jogo interativo do Frozen Lake usando o teclado.
Use as teclas: W (cima), A (esquerda), S (baixo), D (direita) ou setas
Pressione Q para sair
"""
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import sys
import os

# Desabilitar buffer de saída para melhor interatividade
os.environ['PYTHONUNBUFFERED'] = '1'


def get_key_action():
    """Mapeia a tecla pressionada para uma ação no jogo."""
    print("\nAções disponíveis:")
    print("  W ou ↑ = Cima (UP)")
    print("  A ou ← = Esquerda (LEFT)")
    print("  S ou ↓ = Baixo (DOWN)")
    print("  D ou → = Direita (RIGHT)")
    print("  Q = Sair")
    
    action_map = {
        'w': 3,  # UP
        'a': 0,  # LEFT
        's': 1,  # DOWN
        'd': 2,  # RIGHT
        'W': 3,
        'A': 0,
        'S': 1,
        'D': 2,
    }
    
    while True:
        key = input("\nEscolha sua ação: ").strip()
        
        if key.lower() == 'q':
            return None
        
        if key in action_map:
            return action_map[key]
        else:
            print("Tecla inválida! Use W/A/S/D ou Q para sair.")


def print_action_name(action):
    """Retorna o nome da ação."""
    actions = {0: "ESQUERDA ←", 1: "BAIXO ↓", 2: "DIREITA →", 3: "CIMA ↑"}
    return actions.get(action, "DESCONHECIDA")


def play_frozen_lake(map_size=4, is_slippery=False):
    """
    Joga o Frozen Lake interativamente.
    
    Args:
        map_size: Tamanho do mapa (4, 8, etc)
        is_slippery: Se True, o ambiente é escorregadio (mais difícil)
    """
    # Criar o ambiente com mapa gerado
    if map_size == 4:
        # Usar o mapa padrão 4x4
        env = gym.make(
            'FrozenLake-v1',
            is_slippery=is_slippery,
            render_mode='ansi'
        )
    else:
        # Gerar mapa aleatório para outros tamanhos
        custom_map = generate_random_map(size=map_size, p=0.8)
        env = gym.make(
            'FrozenLake-v1',
            desc=custom_map,
            is_slippery=is_slippery,
            render_mode='ansi'
        )
    
    print("=" * 50)
    print("🎮 BEM-VINDO AO FROZEN LAKE! 🎮")
    print("=" * 50)
    print("\nObjetivo: Chegue ao tesouro (G) sem cair nos buracos (H)")
    print("Legenda:")
    print("  S = Start (início)")
    print("  F = Frozen (gelo - seguro)")
    print("  H = Hole (buraco - fim do jogo)")
    print("  G = Goal (objetivo/tesouro)")
    print(f"\nModo: {'Escorregadio 🧊' if is_slippery else 'Normal 👟'}")
    print("=" * 50)
    
    total_games = 0
    wins = 0
    
    while True:
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        total_games += 1
        print(f"\n{'='*50}")
        print(f"🎯 JOGO #{total_games}")
        print(f"{'='*50}")
        print("\nEstado inicial:")
        print(env.render())
        
        while not done:
            # Obter ação do jogador
            action = get_key_action()
            
            if action is None:  # Jogador quer sair
                print("\n👋 Saindo do jogo...")
                print(f"\n📊 ESTATÍSTICAS FINAIS:")
                print(f"  Jogos: {total_games}")
                print(f"  Vitórias: {wins}")
                if total_games > 0:
                    print(f"  Taxa de vitória: {wins/total_games*100:.1f}%")
                env.close()
                return
            
            # Executar ação
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Mostrar resultado
            print(f"\n{'─'*50}")
            print(f"Passo {steps}: {print_action_name(action)}")
            print(f"{'─'*50}")
            print(env.render())
            
            if terminated:
                if reward > 0:
                    wins += 1
                    print("\n🎉 PARABÉNS! Você chegou ao tesouro! 🏆")
                else:
                    print("\n💀 Oh não! Você caiu num buraco!")
                
                print(f"\nEstatísticas do jogo:")
                print(f"  Passos: {steps}")
                print(f"  Recompensa: {total_reward}")
                print(f"\n📊 Vitórias: {wins}/{total_games}")
                
            elif truncated:
                print("\n⏰ Tempo esgotado!")
        
        # Perguntar se quer jogar novamente
        play_again = input("\n🎮 Jogar novamente? (S/N): ").strip().lower()
        if play_again != 's':
            print("\n👋 Obrigado por jogar!")
            print(f"\n📊 ESTATÍSTICAS FINAIS:")
            print(f"  Jogos: {total_games}")
            print(f"  Vitórias: {wins}")
            if total_games > 0:
                print(f"  Taxa de vitória: {wins/total_games*100:.1f}%")
            env.close()
            break


if __name__ == "__main__":
    print("\n🎮 CONFIGURAÇÃO DO JOGO")
    print("=" * 50)
    
    # Escolher tamanho do mapa
    while True:
        try:
            map_size_input = input("\nTamanho do mapa (4, 8, ou Enter para 4): ").strip()
            map_size = int(map_size_input) if map_size_input else 4
            if map_size in [4, 8]:
                break
            print("Por favor, escolha 4 ou 8")
        except ValueError:
            print("Entrada inválida! Use 4 ou 8")
    
    # Escolher modo
    slippery_input = input("\nModo escorregadio? (S/N, Enter para N): ").strip().lower()
    is_slippery = slippery_input == 's'
    
    # Iniciar o jogo
    play_frozen_lake(map_size=map_size, is_slippery=is_slippery)