from engine import type_changer, states_creator, recommender
from flask import Flask, render_template, request
import logging
import secrets


logging.basicConfig(
    level=logging.INFO, filename=f"FrontendWeb.log", encoding="UTF-8", datefmt="%Y-%m-%d %H:%M:%S",
    format="'%(name)s':\n%(levelname)s %(asctime)s --> %(message)s"
)


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Must be constant in production


@app.errorhandler(Exception)
def handle_all_errors(error):
    logging.error("Unexpected error", exc_info=error)
    return f"""<script>
        alert("{error}");
        window.location.href = "/";
    </script>
    """


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and request.form.get('action', None) != 'finish':
        decks_qty = int(request.form['decks_qty'])
        player_cards = [
            player_card for player_card in request.form['player_cards'].upper().split()
            if type_changer.is_cardable(player_card, raise_if_not=True)
        ]
        dealer_card = request.form['dealer_card'].upper().strip()
        type_changer.is_cardable(dealer_card, raise_if_not=True)
        used_cards = [
            used_card for used_card in request.form['used_cards'].upper().split()
            if type_changer.is_cardable(used_card, raise_if_not=True)
        ]

        if 'action' in request.form and request.form['action'] == 'next':
            used_cards.extend(player_cards)
            used_cards.append(dealer_card)
            return render_template('index.html', used_cards=' '.join(used_cards), decks_qty=decks_qty)

        game_state = states_creator.get_game_state(
            tuple(type_changer.str_to_card(card) for card in player_cards),
            type_changer.str_to_card(dealer_card),
            decks_qty,
            tuple(type_changer.str_to_card(card) for card in used_cards)
        )
        recommendation = recommender.get_recommendation(game_state).name

        return render_template(
            'index.html',
            player_cards=' '.join(player_cards), dealer_card=dealer_card,
            recommendation=recommendation, used_cards=' '.join(used_cards), decks_qty=decks_qty,
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)
