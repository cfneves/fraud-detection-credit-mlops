# Casos de Uso — Detecção de Fraude com Machine Learning

> Documento complementar ao README. Descreve situações reais onde este tipo de
> sistema é aplicado, as decisões que ele suporta e o impacto financeiro envolvido.

---

## 1. O que este sistema faz, de verdade

O modelo recebe os dados de uma transação e retorna um número entre 0 e 1 — a probabilidade de ser fraude. Quem define o que fazer com esse número é o banco, não o modelo.

Isso é importante porque o mesmo modelo pode ter comportamentos diferentes dependendo do contexto:

- Um banco focado em clientes de alta renda vai preferir um threshold mais baixo (bloqueia mais, erra menos, aceita mais atrito)
- Uma fintech digital vai preferir um threshold mais alto (aprova mais, aceita algum risco, prioriza experiência do usuário)

O modelo é uma ferramenta. A política de risco é uma decisão de negócio.

---

## 2. Casos de uso por setor

### Bancos tradicionais (Itaú, Bradesco, Caixa, Santander)

**Aprovação de transações no cartão de débito e crédito**
Cada transação passa pelo modelo antes de ser aprovada. O resultado entra numa camada de regras junto com outros sinais (limite disponível, histórico do cliente, lista negra) e a decisão final sai em menos de 300ms.

**Definição de limite dinâmico**
Além de aprovar ou bloquear, o score de risco alimenta a lógica de limite. Um cliente com score de risco crescente pode ter o limite reduzido preventivamente, antes de uma fraude acontecer.

**Triagem para análise manual**
Transações com score entre 30% e 70% vão para uma fila de analistas antifraude. O modelo não decide — ele prioriza. Sem o score, os analistas teriam que revisar todas as transações em ordem de chegada; com o score, começam pelas mais suspeitas.

---

### Fintechs (Nubank, C6, Inter, PicPay)

**Onboarding com validação de identidade**
Na abertura de conta, o modelo analisa padrões de comportamento durante o cadastro: velocidade de preenchimento, dispositivo usado, localização do IP, consistência das informações. Fraude de identidade (usar dados de outra pessoa para abrir conta) é detectada antes do primeiro uso.

**Antifraude em Pix**
O Pix exige decisão em até 10 segundos. O modelo precisa ser leve e rápido — LightGBM foi escolhido por isso. Transações Pix têm padrões diferentes do cartão: valores maiores, horários estendidos, chaves novas como destino são sinais de risco.

**Chargeback preditivo**
Antes de o cliente pedir o estorno, o modelo já identifica transações com alto risco de disputa. O banco pode entrar em contato proativamente e resolver antes de virar chargeback — que custa multas da bandeira além do valor reembolsado.

---

### Adquirentes e processadoras (Cielo, Stone, Rede, GetNet)

**Antifraude no ponto de venda**
A adquirente processa a transação entre o lojista e o banco emissor. O modelo da adquirente analisa o lado do comerciante: equipamento usado, histórico do terminal, padrão de valores, horário de funcionamento.

**Detecção de comerciante fraudulento**
Às vezes o problema não é o comprador — é o lojista. Comerciantes fantasma criam terminais para processar transações de cartões clonados. O modelo detecta padrões como: volume atípico, todos os valores terminam em números redondos, concentração em horário noturno.

**Monitoramento de máquinas clonadas**
Terminais físicos podem ser adulterados para capturar dados de cartão (skimming). O modelo detecta quando um terminal começa a gerar um número anormal de chargebacks nas semanas seguintes às transações.

---

### Bureaus de crédito (Serasa Experian, Boa Vista SCPC)

**Enriquecimento de score de crédito**
O histórico de comportamento antifraude do cliente enriquece o score de crédito. Um cliente que nunca teve suspeita de fraude em 5 anos tem um perfil diferente de alguém com 3 contestações no último trimestre.

**Detecção de fraude cadastral**
CPF com histórico limpo sendo usado para abrir contas em vários bancos no mesmo dia é um sinal claro de fraude de identidade. O bureau cruza informações de múltiplas instituições e o modelo identifica o padrão.

**Score de verificação de identidade (KYC)**
Know Your Customer — o bureau valida se a pessoa que está abrindo a conta é quem diz ser. O modelo analisa consistência entre dados declarados, histórico de crédito, biometria e comportamento digital.

---

### E-commerce (Mercado Livre, Americanas, Shopee)

**Aprovação de pedidos antes do envio**
A fraude em e-commerce geralmente usa cartão clonado para comprar produto e revender. O modelo analisa o pedido antes de confirmar o envio: endereço de entrega diferente do histórico, produto de alto valor sem histórico de compras similares, primeiro pedido de alto valor.

**Detecção de conta laranja**
Contas criadas especificamente para receber produto fraudado. Criadas recentemente, sem histórico de avaliações, endereços de entrega que mudam a cada pedido — o modelo cruza esses sinais.

**Prevenção de abuso de promoções**
Usuários que criam múltiplas contas para usar cupons de desconto "para novos clientes". O comportamento de navegação, dispositivo e padrão de compra ajudam a identificar a mesma pessoa em diferentes contas.

---

## 3. Os cinco tipos de fraude mais comuns

### Card Testing (teste de cartão)
O fraudador obtém um lote de dados de cartão (geralmente via vazamento ou compra na dark web) e precisa saber quais ainda estão ativos. Faz transações muito pequenas — R$1,99, R$0,99 — em serviços de assinatura ou streaming.

**Sinais no modelo:** valor muito baixo fora do padrão histórico, tentativas em sequência rápida, comerciante do tipo "digital goods", horário madrugada.

### Clonagem de cartão físico (Skimming)
Dispositivos instalados em caixas eletrônicos ou maquininhas capturam os dados do cartão na hora da transação. Os dados são vendidos ou usados diretamente.

**Sinais no modelo:** transação em localização geograficamente impossível (o cliente está em São Paulo, a compra aparece em Buenos Aires 1h depois), valor acima da média histórica, comerciante nunca usado antes.

### Account Takeover (invasão de conta)
Criminoso obtém acesso às credenciais do cliente (phishing, credential stuffing) e assume o controle da conta. Antes de fazer compras grandes, costuma mudar dados de cadastro — endereço, telefone, email.

**Sinais no modelo:** login de IP/dispositivo nunca visto, alteração de dados seguida de compra grande num intervalo curto, comportamento de navegação diferente do padrão (velocidade, sequência de cliques).

### Fraude de identidade (Synthetic Identity Fraud)
Combinação de dados reais (CPF verdadeiro) com dados falsos (nome, endereço). A identidade sintética não existe — é criada especificamente para fraude. O criminoso a usa para abrir contas e pegar crédito que nunca será pago.

**Sinais no modelo:** CPF com histórico de crédito inconsistente, dados de contato que nunca foram usados antes, abertura de múltiplas contas em curto período, comportamento de "construção de histórico" antes de pedir crédito alto.

### Friendly Fraud (fraude amiga)
O próprio titular faz uma compra legítima e depois contesta a transação alegando fraude. Responsável por uma parte significativa dos chargebacks no e-commerce.

**Sinais no modelo:** histórico de contestações anteriores, produto de alto valor e fácil revenda, endereço de entrega confirmado, padrão de compra consistente com histórico do cliente. Este é o caso mais difícil — o modelo não consegue distinguir sozinho, precisa de análise humana.

---

## 4. Impacto financeiro

### O custo de uma fraude não detectada

Quando uma fraude passa pelo sistema:
- O banco reembolsa o cliente (valor integral da transação)
- Paga multa para a bandeira (Visa/Mastercard) pelo chargeback
- Absorve o custo operacional de resolver a disputa
- Ainda corre risco de reputação se o cliente tiver uma experiência ruim

Estimativa de mercado: cada fraude não detectada custa ao banco entre 2x e 3x o valor da transação original.

### O custo de um falso alarme

Quando o sistema bloqueia uma transação legítima:
- Custo operacional do atendimento ao cliente
- Atrito: o cliente pode cancelar o cartão ou migrar para outro banco
- Em e-commerce, pode ser a última vez que aquele cliente tenta comprar ali

Estimativa: entre R$5 e R$15 por falso alarme (direto + indireto).

### Por que o threshold importa tanto

Com FN custando 20x mais que FP, o threshold ótimo não é 0.5 — é muito mais baixo. Neste projeto, o threshold foi calculado em **0.1005**, minimizando o custo total esperado, não o F1.

Um banco que usa threshold 0.5 (padrão de livro didático) vai deixar passar muito mais fraude do que precisaria. A diferença entre threshold 0.5 e threshold 0.1 pode representar milhões de reais em fraudes evitadas por mês.

---

## 5. O que o modelo não faz

É importante ser honesto sobre as limitações:

**Não detecta fraude em tempo real de ponta a ponta por si só.** O modelo é uma peça num sistema maior. Precisa de infraestrutura de deploy, monitoramento, regras de negócio e revisão humana para os casos borderline.

**Não substitui análise humana.** Casos de score médio (30-70%) geralmente vão para analistas. O modelo prioriza a fila, não elimina o trabalho humano.

**Degrada ao longo do tempo.** Fraudadores se adaptam. Um modelo treinado hoje vai perder performance em 6-12 meses se não for retreinado com dados novos. O PSI monitora isso, mas não conserta sozinho.

**Não lida bem com fraude completamente nova.** Se um padrão de fraude que nunca existiu antes aparece no mercado, o modelo demora para aprender. Regras manuais dos analistas ainda são necessárias para novos vetores de ataque.

---

## 6. Como este projeto se posiciona como portfólio

Este projeto foi construído para demonstrar fluência nos conceitos que times de dados do mercado financeiro brasileiro efetivamente usam:

- **KS e Gini** são as métricas que o gerente de risco vai pedir, não ROC-AUC
- **Calibração** é o que permite usar a probabilidade para precificação, não só classificação
- **SHAP** é o que resolve a exigência de explicabilidade da LGPD e do BACEN
- **Threshold por custo** é como se toma a decisão em produção, não pelo F1 genérico
- **SMOTE com ratio controlado** mostra que se entende o trade-off, não só que se sabe que a técnica existe
- **Dashboard didático** demonstra capacidade de comunicar resultados técnicos para stakeholders não técnicos

---

*Para explorar o modelo interativamente:*
*[fraud-detection-credit-mlops-cfn.streamlit.app](https://fraud-detection-credit-mlops-cfn.streamlit.app)*
