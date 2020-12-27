Simple implementation of parsing and rolling dice for Dungeons and Dragons and
similar applications.

Sample use:

```
import roll

rolls = roll.get_rolls('d20a * 2 6 4d6k3')
print(roll.rolls_string(rolls))
rolls[0].reroll(1)
print(rolls[0])
```
