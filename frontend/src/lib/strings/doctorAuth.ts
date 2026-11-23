import type { Lang } from "@/lib/i18n";

export interface DoctorAuthStrings {
  title: string;
  subtitle: string;
  tabSignIn: string;
  tabJoin: string;
  username: string;
  usernameHint: string;
  password: string;
  passwordHint: string;
  showPassword: string;
  hidePassword: string;
  signIn: string;
  signingIn: string;
  inviteCode: string;
  inviteCodeHint: string;
  fullName: string;
  specialty: string;
  join: string;
  joining: string;
  demoTitle: string;
  demoText: string;
  demoButton: string;
  demoLoading: string;
  backToPatient: string;
  errWrongCredentials: string;
  errTooMany: string;
  errNetwork: string;
  errGeneric: string;
  errInvalidInvite: string;
  errUsernameTaken: string;
  errDemoUnavailable: string;
  errFieldInvite: string;
  errFieldUsername: string;
  errFieldFullName: string;
  errFieldSpecialty: string;
  errFieldPassword: string;
  errFieldRequired: string;
  // account menu
  account: string;
  demoPill: string;
  invite: string;
  changePassword: string;
  accuracy: string;
  logout: string;
  demoNote: string;
  // invite dialog
  inviteTitle: string;
  inviteGenerating: string;
  inviteCopy: string;
  inviteCopied: string;
  inviteExpires: string;
  inviteRule: string;
  inviteNew: string;
  inviteErr: string;
  close: string;
  // password dialog
  pwTitle: string;
  pwOld: string;
  pwNew: string;
  pwRepeat: string;
  pwSave: string;
  pwSaving: string;
  pwMismatch: string;
  pwWrong: string;
  pwWeak: string;
  pwSuccess: string;
  pwErr: string;
}

export const doctorAuthStrings: Record<Lang, DoctorAuthStrings> = {
  ru: {
    title: "Кабинет врача",
    subtitle: "Войдите, чтобы увидеть очередь пациентов",
    tabSignIn: "Вход",
    tabJoin: "По коду приглашения",
    username: "Логин",
    usernameHint: "Латинские буквы, цифры и . _ -",
    password: "Пароль",
    passwordHint: "От 8 символов, с латинской буквой и цифрой",
    showPassword: "Показать пароль",
    hidePassword: "Скрыть пароль",
    signIn: "Войти",
    signingIn: "Входим…",
    inviteCode: "Код приглашения",
    inviteCodeHint: "Например TMS-7K3Q-M9PX, тире необязательны",
    fullName: "Полное имя",
    specialty: "Специальность",
    join: "Создать аккаунт",
    joining: "Создаём…",
    demoTitle: "Демо-кабинет",
    demoText: "Шесть примерных пациентов, пароль не нужен.",
    demoButton: "Войти как демо-врач",
    demoLoading: "Открываем демо…",
    backToPatient: "Я пациент: на страницу пациента",
    errWrongCredentials: "Неверный логин или пароль.",
    errTooMany: "Слишком много попыток. Подождите минуту и повторите.",
    errNetwork: "Нет связи с сервером. Проверьте интернет и повторите.",
    errGeneric: "Что-то пошло не так. Повторите попытку.",
    errInvalidInvite: "Код неверный, уже использован или истёк. Попросите коллегу выдать новый.",
    errUsernameTaken: "Этот логин занят. Выберите другой.",
    errDemoUnavailable: "Демо-режим сейчас недоступен.",
    errFieldInvite: "Введите код приглашения.",
    errFieldUsername: "Логин: 3-32 символа, латинские буквы, цифры и . _ -",
    errFieldFullName: "Укажите полное имя.",
    errFieldSpecialty: "Укажите специальность.",
    errFieldPassword: "Пароль: от 8 символов, хотя бы одна латинская буква и одна цифра.",
    errFieldRequired: "Заполните все поля.",
    account: "Аккаунт",
    demoPill: "Демо",
    invite: "Пригласить коллегу",
    changePassword: "Сменить пароль",
    accuracy: "Точность модели",
    logout: "Выйти",
    demoNote: "Демо-сессия: изменения не сохраняются в реальных записях.",
    inviteTitle: "Приглашение для коллеги",
    inviteGenerating: "Создаём код…",
    inviteCopy: "Копировать",
    inviteCopied: "Скопировано",
    inviteExpires: "Действует до",
    inviteRule: "Действует 7 дней, срабатывает один раз.",
    inviteNew: "Создать другой код",
    inviteErr: "Не удалось создать код. Повторите попытку.",
    close: "Закрыть",
    pwTitle: "Смена пароля",
    pwOld: "Текущий пароль",
    pwNew: "Новый пароль",
    pwRepeat: "Повторите новый пароль",
    pwSave: "Сохранить",
    pwSaving: "Сохраняем…",
    pwMismatch: "Пароли не совпадают.",
    pwWrong: "Текущий пароль неверный.",
    pwWeak: "Новый пароль: от 8 символов, хотя бы одна латинская буква и одна цифра.",
    pwSuccess: "Пароль изменён.",
    pwErr: "Не удалось сменить пароль. Повторите попытку.",
  },
  kk: {
    title: "Дәрігер кабинеті",
    subtitle: "Науқастар кезегін көру үшін кіріңіз",
    tabSignIn: "Кіру",
    tabJoin: "Шақыру коды бойынша",
    username: "Логин",
    usernameHint: "Латын әріптері, сандар және . _ -",
    password: "Құпиясөз",
    passwordHint: "Кемінде 8 таңба, латын әрпі мен сан болуы керек",
    showPassword: "Құпиясөзді көрсету",
    hidePassword: "Құпиясөзді жасыру",
    signIn: "Кіру",
    signingIn: "Кіріп жатырмыз…",
    inviteCode: "Шақыру коды",
    inviteCodeHint: "Мысалы, TMS-7K3Q-M9PX, сызықшасыз да болады",
    fullName: "Толық аты-жөні",
    specialty: "Мамандығы",
    join: "Аккаунт жасау",
    joining: "Жасап жатырмыз…",
    demoTitle: "Демо-кабинет",
    demoText: "Алты үлгі науқас, құпиясөз керек емес.",
    demoButton: "Демо-дәрігер ретінде кіру",
    demoLoading: "Демо ашылып жатыр…",
    backToPatient: "Мен науқаспын: науқас бетіне өту",
    errWrongCredentials: "Логин немесе құпиясөз қате.",
    errTooMany: "Тым көп әрекет. Бір минут күтіп, қайталаңыз.",
    errNetwork: "Серверге қосылу мүмкін емес. Интернетті тексеріп, қайталаңыз.",
    errGeneric: "Бірдеңе дұрыс болмады. Қайталап көріңіз.",
    errInvalidInvite: "Код қате, қолданылған немесе мерзімі өткен. Әріптесіңізден жаңасын сұраңыз.",
    errUsernameTaken: "Бұл логин бос емес. Басқасын таңдаңыз.",
    errDemoUnavailable: "Демо-режим қазір қолжетімсіз.",
    errFieldInvite: "Шақыру кодын енгізіңіз.",
    errFieldUsername: "Логин: 3-32 таңба, латын әріптері, сандар және . _ -",
    errFieldFullName: "Толық аты-жөніңізді жазыңыз.",
    errFieldSpecialty: "Мамандығыңызды жазыңыз.",
    errFieldPassword: "Құпиясөз: кемінде 8 таңба, кемінде бір латын әрпі және бір сан.",
    errFieldRequired: "Барлық өрісті толтырыңыз.",
    account: "Аккаунт",
    demoPill: "Демо",
    invite: "Әріптесті шақыру",
    changePassword: "Құпиясөзді ауыстыру",
    accuracy: "Модель дәлдігі",
    logout: "Шығу",
    demoNote: "Демо-сессия: өзгерістер нақты жазбаларға сақталмайды.",
    inviteTitle: "Әріптеске шақыру",
    inviteGenerating: "Код жасалып жатыр…",
    inviteCopy: "Көшіру",
    inviteCopied: "Көшірілді",
    inviteExpires: "Жарамдылық мерзімі",
    inviteRule: "7 күн жарамды, бір рет қана істейді.",
    inviteNew: "Басқа код жасау",
    inviteErr: "Код жасау мүмкін болмады. Қайталап көріңіз.",
    close: "Жабу",
    pwTitle: "Құпиясөзді ауыстыру",
    pwOld: "Ағымдағы құпиясөз",
    pwNew: "Жаңа құпиясөз",
    pwRepeat: "Жаңа құпиясөзді қайталаңыз",
    pwSave: "Сақтау",
    pwSaving: "Сақталып жатыр…",
    pwMismatch: "Құпиясөздер сәйкес емес.",
    pwWrong: "Ағымдағы құпиясөз қате.",
    pwWeak: "Жаңа құпиясөз: кемінде 8 таңба, кемінде бір латын әрпі және бір сан.",
    pwSuccess: "Құпиясөз өзгертілді.",
    pwErr: "Құпиясөзді өзгерту мүмкін болмады. Қайталап көріңіз.",
  },
  en: {
    title: "Doctor workspace",
    subtitle: "Sign in to see your patient queue",
    tabSignIn: "Sign in",
    tabJoin: "Join with a code",
    username: "Username",
    usernameHint: "Latin letters, digits and . _ -",
    password: "Password",
    passwordHint: "At least 8 characters, with a Latin letter and a digit",
    showPassword: "Show password",
    hidePassword: "Hide password",
    signIn: "Sign in",
    signingIn: "Signing in…",
    inviteCode: "Invite code",
    inviteCodeHint: "For example TMS-7K3Q-M9PX, dashes are optional",
    fullName: "Full name",
    specialty: "Specialty",
    join: "Create account",
    joining: "Creating…",
    demoTitle: "Demo workspace",
    demoText: "Six example patients, no password needed.",
    demoButton: "Enter as demo doctor",
    demoLoading: "Opening demo…",
    backToPatient: "I am a patient: go to the patient page",
    errWrongCredentials: "Wrong username or password.",
    errTooMany: "Too many attempts. Wait a minute and try again.",
    errNetwork: "Cannot reach the server. Check your connection and try again.",
    errGeneric: "Something went wrong. Please try again.",
    errInvalidInvite: "The code is wrong, already used or expired. Ask a colleague for a new one.",
    errUsernameTaken: "This username is taken. Choose another one.",
    errDemoUnavailable: "Demo mode is not available right now.",
    errFieldInvite: "Enter the invite code.",
    errFieldUsername: "Username: 3-32 characters, Latin letters, digits and . _ -",
    errFieldFullName: "Enter your full name.",
    errFieldSpecialty: "Enter your specialty.",
    errFieldPassword: "Password: at least 8 characters, with a Latin letter and a digit.",
    errFieldRequired: "Fill in all fields.",
    account: "Account",
    demoPill: "Demo",
    invite: "Invite a colleague",
    changePassword: "Change password",
    accuracy: "Model accuracy",
    logout: "Log out",
    demoNote: "Demo session: changes are not saved to real records.",
    inviteTitle: "Invite a colleague",
    inviteGenerating: "Creating code…",
    inviteCopy: "Copy",
    inviteCopied: "Copied",
    inviteExpires: "Valid until",
    inviteRule: "Valid for 7 days, works once.",
    inviteNew: "Create another code",
    inviteErr: "Could not create a code. Please try again.",
    close: "Close",
    pwTitle: "Change password",
    pwOld: "Current password",
    pwNew: "New password",
    pwRepeat: "Repeat new password",
    pwSave: "Save",
    pwSaving: "Saving…",
    pwMismatch: "Passwords do not match.",
    pwWrong: "Current password is wrong.",
    pwWeak: "New password: at least 8 characters, with a Latin letter and a digit.",
    pwSuccess: "Password changed.",
    pwErr: "Could not change the password. Please try again.",
  },
};

export function da(lang: Lang): DoctorAuthStrings {
  return doctorAuthStrings[lang] ?? doctorAuthStrings.ru;
}

export const LOCALE: Record<Lang, string> = { ru: "ru-RU", kk: "kk-KZ", en: "en-GB" };
